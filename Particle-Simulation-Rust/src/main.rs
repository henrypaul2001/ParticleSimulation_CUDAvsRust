use std::{time::Instant, sync::{Mutex, Arc}};
use glium::{Frame, VertexBuffer, Program, IndexBuffer, DrawParameters, glutin::event::{ElementState, VirtualKeyCode}};


use bmp::{Image, Pixel};

#[macro_use]
extern crate glium;

#[derive(Copy, Clone)]
pub struct Vertex {
    position: [f32; 3],
}

#[derive(Copy, Clone)]
pub struct Colour {
    red: f32,
    green: f32,
    blue: f32
}

#[allow(unused_imports)]
use glium::{glutin, Surface};

const GRAVITY: f32 = 9.8;
const DRAG: f32 = 0.05;
const BLEND: f32 = 0.1;
const UPDATE_THREADS: u32 = 30;
const COLLISION_THREADS: u32 = 30;
const SPRAY_THREADS: u32 = 30;
const BATCH_SIZE: u32 = 100;
const TARGET_BATCHES: u32 = 1000;
const RENDERMODE: bool = true;

pub struct Paper {
    paper_array: Vec<Vec<Colour>>,
    particles_missed: i32
    //let mut paper_array = vec![vec![Colour {red: 1.0, green: 1.0, blue: 1.0}; 1000]; 1000];
}

impl Paper {
    pub fn new() -> Paper {
        Paper {
            paper_array: vec![vec![Colour {red: 1.0, green: 1.0, blue: 1.0}; 1000]; 1000],
            particles_missed: 0
        }
    }

    pub fn check_for_collisions(&mut self, particles: &[Particle]) {
        for particle in particles {
            if particle.collided == true {
                self.particle_collided(&particle)
            }
        }
    }

    pub fn particle_collided(&mut self, particle: &Particle) {
        if particle.x > 100.0 || particle.x < 0.0 {
            // Particle missed the paper
            self.particles_missed += 1;
            //println!("Particles missed: {}", self.particles_missed);
        }
        else if particle.z > 100.0 || particle.z < 0.0 {
            // Particle missed the paper
            self.particles_missed += 1;
            //println!("Particles missed: {}", self.particles_missed);
        }
        else {
            let scaled_x: f32 = particle.x * 10.0;
            let scaled_z: f32 = particle.z * 10.0;

            let colour_on_paper: Colour = self.paper_array[scaled_x as usize][scaled_z as usize];
            let new_red: f32 = (1.0 - BLEND) * colour_on_paper.red + (BLEND * particle.colour.red);
            let new_green: f32 = (1.0 - BLEND) * colour_on_paper.green + (BLEND * particle.colour.green);
            let new_blue: f32 = (1.0 - BLEND) * colour_on_paper.blue + (BLEND * particle.colour.blue);

            //println!("Red {} | Green {} | Blue {}", new_red, new_green, new_blue);

            let new_colour: Colour = Colour { red: new_red, green: new_green, blue: new_blue };
            self.paper_array[scaled_x as usize][scaled_z as usize] = new_colour;
        }
    }
}

pub struct Particle {
    colour: Colour,
    x: f32,
    y: f32,
    z: f32,
    velocity_x: f32,
    velocity_y: f32,
    velocity_z: f32,
    collided: bool,
}

impl Particle {
    pub fn new(colour_param: Colour, x_param: f32, y_param: f32, z_param: f32, velocity_x_param: f32, velocity_y_param: f32, velocity_z_param: f32) -> Particle {
        Particle {
            colour: colour_param,
            x: x_param,
            y: y_param,
            z: z_param,
            velocity_x: velocity_x_param,
            velocity_y: velocity_y_param,
            velocity_z: velocity_z_param,
            collided: false,
        }
    }

    pub fn update(&mut self, time_step: f32) {
        if !self.collided {
            // GRAVITY = 9.8
            // DRAG = 0.05
            let acceleration_x = -DRAG * self.velocity_x.powi(2);
            let distance_x = self.velocity_x * time_step + 0.5 * acceleration_x * time_step.powi(2);

            let acceleration_y = GRAVITY -DRAG * self.velocity_y.powi(2);
            let distance_y = self.velocity_y * time_step + 0.5 * acceleration_y * time_step.powi(2);

            let acceleration_z = -DRAG * self.velocity_z.powi(2);
            let distance_z = self.velocity_z * time_step + 0.5 * acceleration_z * time_step.powi(2);

            self.x += distance_x;
            self.y += distance_y;
            self.z += distance_z;

            self.y = self.y.clamp(0.0, 100000000.0);

            if self.velocity_x.is_sign_negative() {
                self.velocity_x += -acceleration_x * time_step;
            }
            else {
                self.velocity_x += acceleration_x * time_step;
            }

            self.velocity_y += -acceleration_y * time_step;

            if self.velocity_z.is_sign_negative() {
                self.velocity_z += -acceleration_z * time_step;
            }
            else {
                self.velocity_z += acceleration_z * time_step;
            }

            // Collision
            if self.y == 0.0 {
                self.collided = true;
            }
        }
    }
}

pub struct AerosolCan {
    particles: Vec<Particle>,
    x: f32,
    y: f32,
    z: f32,
    base_velocity_x: f32,
    base_velocity_y: f32,
    base_velocity_z: f32,
    colour: Colour,
    spray_radius: f32,
    particles_created: u32,
}

impl AerosolCan {
    pub fn new(colour_param: Colour, x_param: f32, y_param: f32, z_param: f32, velocity_x_param: f32, velocity_y_param: f32, velocity_z_param: f32, radius_param: f32) -> AerosolCan {
        AerosolCan { 
            particles: Vec::new(),
            colour: colour_param,
            x: x_param,
            y: y_param,
            z: z_param,
            base_velocity_x: velocity_x_param,
            base_velocity_y: velocity_y_param,
            base_velocity_z: velocity_z_param,
            spray_radius: radius_param,
            particles_created: 0
        }
    }

    pub fn update_particles(&mut self, time_step: f32) {
        for particle in &mut self.particles {
            if !particle.collided {
                particle.update(time_step);
            }
        }
    }

    pub fn print_particles(&self) {
        let mut index = 1;
        for particle in &self.particles {
            if index < 500 {
                println!("Particle {} | X: {} | Y: {} | Z: {}", index, particle.x, particle.y, particle.z);
            }
            index += 1;
        }
        println!("");
    }

    pub fn spray(&mut self, number_of_particles: u32) {
        let mut radius: f32 = self.spray_radius;
        let mut particles = number_of_particles;

        while radius > 0.0 && particles > 0 {
            // Create new particles
            for i in 0..particles {
                let horizontal_angle = i as f32 / particles as f32 * 2.0 * std::f32::consts::PI;
                let vertical_angle = i as f32 / particles as f32 * std::f32::consts::PI;
                
                //let angle = i as f32 * (std::f32::consts::PI / (particles as f32 / 2.0));
                let slice = 2.0 * std::f32::consts::PI / particles as f32;
                let angle = slice * i as f32;

                let mut new_x = self.x;
                let mut new_y = self.y;
                let mut new_z = self.z;
                if self.base_velocity_z == 0.0 {
                    new_y = self.y + radius * angle.sin();
                    new_z = self.z + radius * angle.cos();

                }
                else if self.base_velocity_x == 0.0 {
                    new_x = self.x + radius * angle.cos();
                    new_y = self.y + radius * angle.sin();
                }
                let new_x = self.x + radius * horizontal_angle.cos() * vertical_angle.sin();
                let new_y = self.y + radius * horizontal_angle.sin() * vertical_angle.sin();
                let new_z = self.z + radius * vertical_angle.cos();

                self.particles.push(Particle::new(self.colour, new_x, new_y, new_z, 
                    self.base_velocity_x, self.base_velocity_y, self.base_velocity_z));
                self.particles_created += 1;
            }

            radius = radius / 1.25;
            particles = particles / 2;
        }
    }
}

fn main() {
/*
    // Create particles
    let particle_count = 3000;
    for _i in 0..particle_count {
        let rand_x = 0.0;//rand::thread_rng().gen_range(-1.0..=1.0);
        let rand_y = rand::thread_rng().gen_range(-1.0..=1.0);
        let rand_z = rand::thread_rng().gen_range(-1.0..=1.0);

        let rand_x_vel = rand::thread_rng().gen_range(0.0..=5.0);
        let rand_y_vel = rand::thread_rng().gen_range(0.0..=3.0);
        let rand_z_vel = rand::thread_rng().gen_range(-3.0..=3.0);
        red_particles.push(Particle::new(1.0, 0.0, 0.0, red_base_x + rand_x, red_base_y + rand_y, red_base_z + rand_z,
            red_base_x_vel + rand_x_vel, red_base_y_vel + rand_y_vel, red_base_z_vel + rand_z_vel));
    }

    for _i in 0..particle_count {
        let rand_x = 0.0;//rand::thread_rng().gen_range(-1.0..=1.0);
        let rand_y = rand::thread_rng().gen_range(-1.0..=1.0);
        let rand_z = rand::thread_rng().gen_range(-1.0..=1.0);

        let rand_x_vel = rand::thread_rng().gen_range(-5.0..=0.0);
        let rand_y_vel = rand::thread_rng().gen_range(0.0..=3.0);
        let rand_z_vel = rand::thread_rng().gen_range(-3.0..=3.0);
        green_particles.push(Particle::new(0.0, 1.0, 0.0, green_base_x + rand_x, green_base_y + rand_y, green_base_z + rand_z,
            green_base_x_vel + rand_x_vel, green_base_y_vel + rand_y_vel, green_base_z_vel + rand_z_vel));
    }

    for _i in 0..particle_count {
        let rand_x = rand::thread_rng().gen_range(-1.0..=1.0);
        let rand_y = rand::thread_rng().gen_range(-1.0..=1.0);
        let rand_z = 0.0;//rand::thread_rng().gen_range(-1.0..=1.0);

        let rand_x_vel = rand::thread_rng().gen_range(-3.0..=3.0);
        let rand_y_vel = rand::thread_rng().gen_range(0.0..=3.0);
        let rand_z_vel = rand::thread_rng().gen_range(-5.0..=0.0);
        blue_particles.push(Particle::new(0.0, 0.0, 1.0, blue_base_x + rand_x, blue_base_y + rand_y, blue_base_z + rand_z,
            blue_base_x_vel + rand_x_vel, blue_base_y_vel + rand_y_vel, blue_base_z_vel + rand_z_vel));
    }
    */
    //let spray_time = 5.0;
    let paper = Arc::new(Mutex::new(Paper::new()));

    let red_can = Arc::new(Mutex::new(AerosolCan::new(Colour { red: 1.0, green: 0.0, blue: 0.0 }, -15.0, 30.0, 60.0, 125.0, 0.0, 0.0, 10.0)));
    let green_can = Arc::new(Mutex::new(AerosolCan::new(Colour { red: 0.0, green: 1.0, blue: 0.0 }, 115.0, 30.0, 60.0, -125.0, 0.0, 0.0, 10.0)));
    let blue_can = Arc::new(Mutex::new(AerosolCan::new(Colour { red: 0.0, green: 0.0, blue: 1.0 }, 50.0, 30.0, 125.0, 0.0, 0.0, -125.0, 10.0)));

    // OpenGL stuff
    let event_loop = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new();
    let cb = glutin::ContextBuilder::new().with_depth_buffer(24);
    let display = glium::Display::new(wb, cb, &event_loop).unwrap();

    implement_vertex!(Vertex, position);

    // Cube for particles
    let vertex1 = Vertex { position: [ -1.0, -1.0,  1.0 ] };
    let vertex2 = Vertex { position: [ 1.0, -1.0,  1.0 ] };
    let vertex3 = Vertex { position: [ -1.0, 1.0, 1.0 ] };
    let vertex4 = Vertex { position: [ 1.0, 1.0, 1.0 ] };
    let vertex5 = Vertex { position: [ -1.0, -1.0, -1.0 ] };
    let vertex6 = Vertex { position: [ 1.0, -1.0, -1.0 ] };
    let vertex7 = Vertex { position: [ -1.0, 1.0, -1.0 ] };
    let vertex8 = Vertex { position: [ 1.0, 1.0, -1.0 ] };
    let shape = vec![vertex1, vertex2, vertex3, vertex4, vertex5, vertex6, vertex7, vertex8];

    let cube_indices: [u16; 36] =
    [
        // Top
        2, 6, 7,
        2, 3, 7,

        // Bottom
        0, 4, 5,
        0, 1, 5,

        // Left
        0, 2, 6,
        0, 4, 6,

        // Right
        1, 3, 7,
        1, 5, 7,

        // Front
        0, 2, 3,
        0, 1, 3,

        // Back
        4, 6, 7,
        4, 5, 7
    ];

    let vertex_buffer = glium::VertexBuffer::new(&display, &shape).unwrap();
    let indices = glium::IndexBuffer::new(&display, glium::index::PrimitiveType::TrianglesList, &cube_indices).unwrap();

    let paper_vertex1 = Vertex { position: [ 0.0, 0.0, 0.0 ] };
    let paper_vertex2 = Vertex { position: [ 100.0, 0.0, 0.0 ] };
    let paper_vertex3 = Vertex { position: [ 100.0, 0.0, 100.0 ] };
    let paper_vertex4 = Vertex { position: [ 0.0, 0.0, 100.0 ] };
    let paper_shape = vec![paper_vertex1, paper_vertex3, paper_vertex4, paper_vertex1, paper_vertex2, paper_vertex3];

    let paper_vertex_buffer = glium::VertexBuffer::new(&display, &paper_shape).unwrap();
    let paper_indices = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);

    let vertex_shader_src = r#"
        #version 140

        in vec3 position;
        out vec3 inColor;

        uniform mat4 model;
        uniform mat4 perspective;
        uniform mat4 view;

        uniform float red;
        uniform float green;
        uniform float blue;

        void main() {
            inColor = vec3(red, green, blue);
            mat4 modelview = view * model;
            gl_Position = perspective * modelview * vec4(position, 1.0);
        }
    "#;

    let fragment_shader_src = r#"
        #version 140

        in vec3 inColor;
        out vec4 color;

        void main() {
            color = vec4(inColor, 1.0);
        }
    "#;

    let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None).unwrap();

    let start_time = Instant::now();
    let mut prev_frame_time = Instant::now();
    let mut simulation_running = true;
    let mut bitmap_created = false;
    let mut current_batches: u32 = 0;
    //let mut RENDERMODE = false;
    event_loop.run(move |event, _, control_flow| {

        match event {
            glutin::event::Event::WindowEvent { event, .. } => match event {
                glutin::event::WindowEvent::CloseRequested => {
                    *control_flow = glutin::event_loop::ControlFlow::Exit;
                    return;
                },
                _ => return,
            },
            glutin::event::Event::NewEvents(cause) => match cause {
                glutin::event::StartCause::ResumeTimeReached { .. } => (),
                glutin::event::StartCause::Init => (),
                _ => return,
            },
            _ => return,
        }

        let next_frame_time = std::time::Instant::now() + std::time::Duration::from_nanos(16_666_667);
        *control_flow = glutin::event_loop::ControlFlow::WaitUntil(next_frame_time);
        //*control_flow = glutin::event_loop::ControlFlow::Poll;

        // Begin render loop
        let current_time = Instant::now();
        let delta_t = current_time.duration_since(prev_frame_time).as_secs_f32() as f32;
        let elapsed_time = current_time.duration_since(start_time).as_secs_f32();
        prev_frame_time = current_time;

        //if simulation_running { println!("FPS: {}", 1.0 / delta_t); }

        // Create a drawing target
        let mut target = display.draw();

        // Clear the screen to black
        target.clear_color_and_depth((0.0, 0.0, 0.0, 1.0), 1.0);

        // Camera and projection
        let position: [f32; 3] = [50.0, 30.0, -60.0];
        let direction: [f32; 3] = [0.0, 0.0, 1.0];
        let up: [f32; 3] = [0.0, 1.0, 0.0];
        let view = view_matrix(&position, &direction, &up);

        let perspective = {
            let (width, height) = target.get_dimensions();
            let aspect_ratio = height as f32 / width as f32;
    
            let fov: f32 = 3.141592 / 3.0;
            let zfar = 1024.0;
            let znear = 0.1;
    
            let f = 1.0 / (fov / 2.0).tan();
            [
                [f *   aspect_ratio   ,    0.0,              0.0              ,   0.0],
                [         0.0         ,     f ,              0.0              ,   0.0],
                [         0.0         ,    0.0,  (zfar+znear)/(zfar-znear)    ,   1.0],
                [         0.0         ,    0.0, -(2.0*zfar*znear)/(zfar-znear),   0.0],
            ]
        };

        let params = glium::DrawParameters {
            depth: glium::Depth {
                test: glium::draw_parameters::DepthTest::IfLess,
                write: true,
                ..Default::default()
            },
            ..Default::default()
        };
        
        //if elapsed_time > spray_time + 5.0 { simulation_running = false; }
        if current_batches >= TARGET_BATCHES {
            simulation_running = false;
        }

        if simulation_running {
            // Spray cans
            let mut spray_pools = scoped_threadpool::Pool::new(SPRAY_THREADS);
            spray_pools.scoped(|scope| {
                for can in vec![red_can.clone(), green_can.clone(), blue_can.clone()] {
                    //let mut can_clone = can.clone();
                    scope.execute(move || spray_can_thread(can, BATCH_SIZE));
                    current_batches += 1;
                }
            });

            red_can.lock().unwrap().x -= 0.25;

            // Update particles
            for can in vec![red_can.clone(), green_can.clone(), blue_can.clone()] {
                let mut update_pool = scoped_threadpool::Pool::new(UPDATE_THREADS);
                let particles = &mut can.lock().unwrap().particles;

                update_pool.scoped(|scope| {
                    for chunk in particles.chunks_mut(BATCH_SIZE as usize) {
                        scope.execute(move || move_particles_thread(chunk, delta_t));
                    }
                })
            }

            // Check for collisions
            for can in vec![red_can.clone(), green_can.clone(), blue_can.clone()] {
                let mut collisions_pool = scoped_threadpool::Pool::new(COLLISION_THREADS);
                let particles = &can.lock().unwrap().particles;

                collisions_pool.scoped(|scope| {
                    for chunk in particles.chunks(BATCH_SIZE as usize) {
                        let paper_clone = paper.clone();
                        scope.execute(move || check_for_collisions_thread(chunk, paper_clone));
                    }
                })
            }
        }

        
        if RENDERMODE {
            for can in vec![red_can.clone(), green_can.clone(), blue_can.clone()] {
                let particles = &can.lock().unwrap().particles;

                for chunk in particles.chunks(BATCH_SIZE as usize) {
                    for particle in chunk {
                        if !particle.collided {        
                            let model_matrix = [
                                [0.1, 0.0, 0.0, 0.0],
                                [0.0, 0.1, 0.0, 0.0],
                                [0.0, 0.0, 0.1, 0.0],
                                [particle.x, particle.y, particle.z, 1.0f32]
                            ];
                
                            // Draw the triangle
                            target.draw(&vertex_buffer, &indices, &program, &uniform! { model: model_matrix, perspective: perspective, view: view, red: particle.colour.red, green: particle.colour.green, blue: particle.colour.blue },
                                 &params).unwrap();
                        }
                    }
                }
            }

            let model_matrix = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0f32]
            ];

            let red: f32 = 1.0;
            let green: f32 = 1.0;
            let blue: f32 = 1.0;

            // Render paper
            target.draw(&paper_vertex_buffer, &paper_indices, &program, &uniform! { model: model_matrix, perspective: perspective, view: view, red: red, green: green, blue: blue }, &params).unwrap();
        }

        // Display the completed drawing
        target.finish().unwrap();

        if !simulation_running && !bitmap_created {
            bitmap_created = true;
            output_bitmap(paper.clone());
            println!("Time to complete: {}", elapsed_time);
            println!("Particles missed: {}", paper.lock().unwrap().particles_missed);
            let total_particles_created = red_can.lock().unwrap().particles_created + green_can.lock().unwrap().particles_created + blue_can.lock().unwrap().particles_created;
            println!("Particls hit: {}", total_particles_created - paper.lock().unwrap().particles_missed as u32);
            println!("Particles created: {}", total_particles_created);
            //red_can.lock().unwrap().print_particles();
        }
        // End render loop
    });
}

pub fn view_matrix(position: &[f32; 3], direction: &[f32; 3], up: &[f32; 3]) -> [[f32; 4]; 4] {
    let f = {
        let f = direction;
        let len = f[0] * f[0] + f[1] * f[1] + f[2] * f[2];
        let len = len.sqrt();
        [f[0] / len, f[1] / len, f[2] / len]
    };

    let s = [up[1] * f[2] - up[2] * f[1],
             up[2] * f[0] - up[0] * f[2],
             up[0] * f[1] - up[1] * f[0]];

    let s_norm = {
        let len = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
        let len = len.sqrt();
        [s[0] / len, s[1] / len, s[2] / len]
    };

    let u = [f[1] * s_norm[2] - f[2] * s_norm[1],
             f[2] * s_norm[0] - f[0] * s_norm[2],
             f[0] * s_norm[1] - f[1] * s_norm[0]];

    let p = [-position[0] * s_norm[0] - position[1] * s_norm[1] - position[2] * s_norm[2],
             -position[0] * u[0] - position[1] * u[1] - position[2] * u[2],
             -position[0] * f[0] - position[1] * f[1] - position[2] * f[2]];

    [
        [s_norm[0], u[0], f[0], 0.0],
        [s_norm[1], u[1], f[1], 0.0],
        [s_norm[2], u[2], f[2], 0.0],
        [p[0], p[1], p[2], 1.0],
    ]
}

pub fn output_bitmap(paper: Arc<Mutex<Paper>>) {
    let paper = paper.lock().unwrap();
    let width = paper.paper_array.len();
    let height = paper.paper_array[0].len();

    let mut image = Image::new(width as u32, height as u32);

    for x in 0..width {
        for y in 0..height {
            let array_x = x;
            let array_y = 999 - y;
            let colour = paper.paper_array[array_x][array_y];
            let pixel = Pixel::new((colour.red * 255.0) as u8, (colour.green * 255.0) as u8, (colour.blue * 255.0) as u8);
            image.set_pixel(x as u32, y as u32, pixel)
        }
    }

    // Save result
    image.save("output.bmp").unwrap();

    println!("Bitmap exported");
}

pub fn spray_can_thread(can: Arc<Mutex<AerosolCan>>, number_of_particles: u32) {
    can.lock().unwrap().spray(number_of_particles);
}

pub fn move_particles_thread(chunk: &mut [Particle], time_step: f32) {
    for particle in chunk {
        particle.update(time_step);
    }
}

pub fn check_for_collisions_thread(chunk: &[Particle], paper: Arc<Mutex<Paper>>) {
    let mut paper = paper.lock().unwrap();
    paper.check_for_collisions(chunk);
}

pub fn render_particles_thread(chunk: &[Particle], target: &mut Frame, vertex_buffer: &VertexBuffer<Vertex>, indices: &IndexBuffer<u16>, program: &Program, perspective: &[[f32; 4]; 4], view: &[[f32; 4]; 4], params: &DrawParameters) {
    for particle in chunk {
        if !particle.collided {        
            let model_matrix = [
                [0.1, 0.0, 0.0, 0.0],
                [0.0, 0.1, 0.0, 0.0],
                [0.0, 0.0, 0.1, 0.0],
                [particle.x, particle.y, particle.z, 1.0f32]
            ];

            // Draw the triangle
            target.draw(vertex_buffer, indices, program, &uniform! { model: model_matrix, perspective: *perspective, view: *view, red: particle.colour.red, green: particle.colour.green, blue: particle.colour.blue }, params).unwrap();
        }
    }
}