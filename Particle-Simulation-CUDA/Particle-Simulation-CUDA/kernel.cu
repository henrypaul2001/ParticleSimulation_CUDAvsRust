#define _USE_MATH_DEFINES
#include <iostream>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>
#include <thread>
#include <stdio.h>
#include <vector>
#include <windows.h>
#include <opencv2/opencv.hpp>

struct Colour {
    float red = 1;
    float green = 1;
    float blue = 1;
};

struct Particle {
    Colour colour = Colour();
    float x = 0;
    float y = 0;
    float z = 0;
    float velocity_x = 0;
    float velocity_y = 0;
    float velocity_z = 0;
    bool collided = false;
    bool landed_on_paper = false;
};

const int max_particles_per_can = 99999;

__global__ void check_for_collision_GPU(Colour* paper_array, Particle* particles, uint32_t* particles_missed, int number_of_particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > number_of_particles) {
        return;
    }

    Particle* particle;
    if (&particles[i] == NULL) {
        return;
    }
    else {
        particle = &particles[i];
    }

    if (particle->collided) {
        if (particle->x > 100.0 || particle->x < 0.0) {
            // Particle missed the paper
            *particles_missed += 1;
            //printf("particle missed");
        }
        else if (particle->z > 100.0 || particle->z < 0.0) {
            // Paricle missed the paper
            *particles_missed += 1;
            //printf("particle missed");
        }
        else {
            if (!particle->landed_on_paper) {
                particle->landed_on_paper = true;
                float scaled_x = particle->x * 10;
                float scaled_z = particle->z * 10;

                int index_x = static_cast<int>(scaled_x);
                int index_z = static_cast<int>(scaled_z);

                const float BLEND = 0.1;
                Colour colour_on_paper = paper_array[index_x * 1000 + index_z];
                float new_red = (1.0 - BLEND) * colour_on_paper.red + (BLEND * particle->colour.red);
                float new_green = (1.0 - BLEND) * colour_on_paper.green + (BLEND * particle->colour.green);
                float new_blue = (1.0 - BLEND) * colour_on_paper.blue + (BLEND * particle->colour.blue);

                Colour new_colour;
                new_colour.red = new_red;
                new_colour.green = new_green;
                new_colour.blue = new_blue;

                paper_array[index_x * 1000 + index_z] = new_colour;
            }
        }
    }
}

__global__ void update_particles_GPU(Particle* particles, const float time_step, const uint32_t particles_created) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("index : (%d)", i);
    //printf("check 0");
    //printf(" created: (%d)", particles_created);
    if (i > particles_created) {
        return;
    }
    //printf("check1");

    Particle* particle;
    if (&particles[i] == NULL) {
        return;
    }
    else {
        particle = &particles[i];
    }
    //printf("check2");
    if (!particle->collided) {
        float GRAVITY = 9.8;
        float DRAG = 0.05;
        float acceleration_x = -DRAG * (particle->velocity_x * particle->velocity_x);
        float distance_x = particle->velocity_x * time_step + 0.5 * acceleration_x * (time_step * time_step);

        float acceleration_y = GRAVITY - DRAG * (particle->velocity_y * particle->velocity_y);
        float distance_y = particle->velocity_y * time_step + 0.5 * acceleration_y * (time_step * time_step);

        float acceleration_z = -DRAG * (particle->velocity_z * particle->velocity_z);
        float distance_z = particle->velocity_z * time_step + 0.5 * acceleration_z * (time_step * time_step);

        particle->x += distance_x;
        particle->y += distance_y;
        particle->z += distance_z;

        if (particle->y < 0) {
            particle->y = 0;
        }

        if (particle->velocity_x < 0) {
            particle->velocity_x += -acceleration_x * time_step;
        }
        else {
            particle->velocity_x += acceleration_x * time_step;
        }

        particle->velocity_y += -acceleration_y * time_step;

        if (particle->velocity_z < 0) {
            particle->velocity_z += -acceleration_z * time_step;
        }
        else {
            particle->velocity_z += acceleration_z * time_step;
        }

        // Collision
        if (particle->y == 0) {
            particle->collided = true;
        }
    }
    //printf("check3");
}

struct AerosolCan {
    Particle* particles = new Particle[max_particles_per_can];
    Particle* device_particles;

    float x = 0;
    float y = 0;
    float z = 0;
    float base_velocity_x = 0;
    float base_velocity_y = 0;
    float base_velocity_z = 0;
    Colour colour = Colour();
    float spray_radius = 0;
    uint32_t particles_created = 0;

    void allocate_memory() {
        Particle* dev_particles = nullptr;

        cudaError_t cuda_status = cudaMalloc((void**)&dev_particles, max_particles_per_can * sizeof(Particle));
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed");
            cudaFree(dev_particles);
        }

        device_particles = dev_particles;
    }

    void update_particles(const float time_step, int num_blocks) {
        // Copy data to GPU
        cudaError_t cuda_status = cudaMemcpy(device_particles, particles, max_particles_per_can * sizeof(Particle), cudaMemcpyHostToDevice);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed: (%d)", cudaGetErrorString(cudaGetLastError()));
            cudaFree(device_particles);
        }

        // Update data on GPU
        int block_size = particles_created / num_blocks;
        //std::cout << "Blocks: " << num_blocks << " | Block size: " << block_size << std::endl;
        update_particles_GPU << <num_blocks, block_size >> > (device_particles, time_step, particles_created);

        // Wait for all threads to exit
        cuda_status = cudaDeviceSynchronize();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "cudaSync failed: (%d)", cudaGetErrorString(cudaGetLastError()));
            cudaFree(device_particles);
        }

        // Copy data back to host
        cuda_status = cudaMemcpy(particles, device_particles, max_particles_per_can * sizeof(Particle), cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed: (%d)", cudaGetErrorString(cudaGetLastError()));
            cudaFree(device_particles);
        }
    }

    void print_particles() {
        int index = 1;
        
        for (int i = 0; i < particles_created; i++) {
            Particle* particle = &particles[i];
            std::cout << "Particle " << i + 1 << " | X: " << particle->x << " | Y: "
                << particle->y << " | Z: " << particle->z << " | Hit = " << particle->landed_on_paper << std::endl;
        }
        
        std::cout << "" << std::endl;
    }

    void spray(uint32_t number_of_particles) {
        float radius = spray_radius;
        while (radius > 0.0 && number_of_particles > 0) {
            // Create new particles
            for (int i = 0; i < number_of_particles; i++) {
                if (particles_created < max_particles_per_can) {
                    float horizontal_angle = i / number_of_particles * 2.0 * M_PI;
                    float vertical_angle = i / number_of_particles * M_PI;

                    float slice = 2.0 * M_PI / number_of_particles;
                    float angle = slice * i;

                    float new_x = x;
                    float new_y = y;
                    float new_z = z;

                    if (base_velocity_z == 0) {
                        new_y = y + radius * sin(angle);
                        new_z = z + radius * cos(angle);
                    }
                    else if (base_velocity_x == 0) {
                        new_x = x + radius * cos(angle);
                        new_y = y + radius * sin(angle);
                    }
                    //new_x = x + radius * cos(horizontal_angle) * sin(vertical_angle);
                    //new_y = y + radius * sin(horizontal_angle) * sin(vertical_angle);
                    //new_z = z + radius * cos(vertical_angle);

                    Particle new_particle = Particle();
                    new_particle.colour = colour;
                    new_particle.x = new_x;
                    new_particle.y = new_y;
                    new_particle.z = new_z;
                    new_particle.velocity_x = base_velocity_x;
                    new_particle.velocity_y = base_velocity_y;
                    new_particle.velocity_z = base_velocity_z;

                    //reinterpret_cast<Particle*>(&particles)[particles_created] = new_particle;
                    particles[particles_created] = new_particle;

                    particles_created++;
                }
            }

            radius = radius / 2.0;
            number_of_particles = number_of_particles / 2.0;
        }
    }
};

struct Paper {
    Colour* paper_array = new Colour[1000 * 1000]();
    uint32_t particles_missed = 0;
    uint32_t* device_particles_missed;
    Colour* device_paper_array;

    void allocate_memory() {
        Colour* dev_paper = nullptr;

        cudaError_t cuda_status = cudaMalloc((void**)&dev_paper, (1000 * 1000) * sizeof(Colour));
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed");
            cudaFree(dev_paper);
        }

        device_paper_array = dev_paper;

        uint32_t* dev_missed = nullptr;

        cuda_status = cudaMalloc((void**)&dev_missed, sizeof(uint32_t));
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed");
            cudaFree(dev_missed);
        }

        device_particles_missed = dev_missed;
    }

    void check_for_collisions(Particle* device_particles, Particle* particles, int number_of_particles, int num_blocks) {

        // Copy data to GPU
        cudaError_t cuda_status = cudaMemcpy(device_paper_array, paper_array, (1000 * 1000) * sizeof(Colour), cudaMemcpyHostToDevice);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed: (%d)", cudaGetErrorString(cudaGetLastError()));
            cudaFree(device_paper_array);
        }

        cuda_status = cudaMemcpy(device_particles_missed, &particles_missed, sizeof(uint32_t), cudaMemcpyHostToDevice);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed: (%d)", cudaGetErrorString(cudaGetLastError()));
            cudaFree(device_paper_array);
        }

        cuda_status = cudaMemcpy(device_particles, particles, max_particles_per_can * sizeof(Particle), cudaMemcpyHostToDevice);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed: (%d)", cudaGetErrorString(cudaGetLastError()));
            cudaFree(device_particles);
        }

        // Check for collisions on GPU
        int block_size = number_of_particles / num_blocks;
        check_for_collision_GPU << <num_blocks, block_size >> > (device_paper_array, device_particles, device_particles_missed, number_of_particles);

        // Wait for all threads to exit
        cuda_status = cudaDeviceSynchronize();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "cudaSync failed: (%d)", cudaGetErrorString(cudaGetLastError()));
            cudaFree(device_particles);
            cudaFree(device_paper_array);
            cudaFree(device_particles_missed);
        }

        // Copy data back to host
        cuda_status = cudaMemcpy(paper_array, device_paper_array, (1000 * 1000) * sizeof(Colour), cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed: (%d)", cudaGetErrorString(cudaGetLastError()));
            cudaFree(device_paper_array);
        }

        cuda_status = cudaMemcpy(&particles_missed, device_particles_missed, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed: (%d)", cudaGetErrorString(cudaGetLastError()));
            cudaFree(device_particles_missed);
        }

        cuda_status = cudaMemcpy(particles, device_particles, max_particles_per_can * sizeof(Particle), cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed: (%d)", cudaGetErrorString(cudaGetLastError()));
            cudaFree(device_particles);
        }
    }
};

void create_bitmap(Colour* paper_array, int array_size_x, int array_size_y, std::string& output_path) {
    // Create an OpenCV matrix for the image
    cv::Mat image(array_size_x, array_size_y, CV_8UC3);

    for (int i = 0; i < array_size_y; i++) {
        for (int j = 0; j < array_size_x; j++) {
            int scaled_i = (array_size_y - 1) - i;
            cv::Vec3b& pixel = image.at<cv::Vec3b>(i, j);
            pixel[0] = static_cast<unsigned char>(paper_array[j * array_size_y + scaled_i].blue * 255.0f);
            pixel[1] = static_cast<unsigned char>(paper_array[j * array_size_y + scaled_i].green * 255.0f);
            pixel[2] = static_cast<unsigned char>(paper_array[j * array_size_y + scaled_i].red * 255.0f);
        }
    }

    // Convert the image to 8-bit representation
    cv::Mat image_8bit;
    image.convertTo(image_8bit, CV_8UC3, 255.0);

    // Output bitmap
    cv::imwrite(output_path, image);
}

const double target_frame_time = 1.0 / 60.0;

int main() {
    Paper paper = Paper();
    #pragma region Cans
    Colour red_colour;
    red_colour.red = 1;
    red_colour.green = 0;
    red_colour.blue = 0;

    
    Colour green_colour;
    green_colour.red = 0;
    green_colour.green = 1;
    green_colour.blue = 0;

    Colour blue_colour;
    blue_colour.red = 0;
    blue_colour.green = 0;
    blue_colour.blue = 1;
    
    AerosolCan red_can = AerosolCan();
    red_can.colour = red_colour;
    red_can.x = -15;
    red_can.y = 30;
    red_can.z = 60;
    red_can.base_velocity_x = 125;
    red_can.base_velocity_y = 0;
    red_can.base_velocity_z = 0;
    red_can.spray_radius = 10;
    red_can.particles = new Particle[max_particles_per_can];
    
    AerosolCan green_can = AerosolCan();
    green_can.colour = green_colour;
    green_can.x = 115;
    green_can.y = 30;
    green_can.z = 60;
    green_can.base_velocity_x = -125;
    green_can.base_velocity_y = 0;
    green_can.base_velocity_z = 0;
    green_can.spray_radius = 10;
    green_can.particles = new Particle[max_particles_per_can];

    AerosolCan blue_can = AerosolCan();
    blue_can.colour = blue_colour;
    blue_can.x = 50;
    blue_can.y = 30;
    blue_can.z = 125;
    blue_can.base_velocity_x = 0;
    blue_can.base_velocity_y = 0;
    blue_can.base_velocity_z = -125;
    blue_can.spray_radius = 10;
    blue_can.particles = new Particle[max_particles_per_can];

    AerosolCan* cans[3];
    cans[0] = &red_can;
    cans[1] = &green_can;
    cans[2] = &blue_can;
    #pragma endregion
    
    // Allocate memory
    red_can.allocate_memory();
    green_can.allocate_memory();
    blue_can.allocate_memory();
    paper.allocate_memory();

    int block_size = 142;
    uint32_t target_batches = 1000;
    uint32_t batch_size = 100;
    uint32_t current_batches = 0;
    cudaError_t cuda_status;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto previous_frame_time = start_time;
    while (current_batches < target_batches) {
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> time_step = current_time - previous_frame_time;
        //system("cls");

        // Check for collisions on GPU
        paper.check_for_collisions(red_can.device_particles, red_can.particles, red_can.particles_created, block_size);
        paper.check_for_collisions(green_can.device_particles, green_can.particles, green_can.particles_created, block_size);
        paper.check_for_collisions(blue_can.device_particles, blue_can.particles, blue_can.particles_created, block_size);

        // Spray cans
        bool all_cans_maxed = true;
        for (AerosolCan* can : cans) {
            if (can->particles_created < max_particles_per_can) {
                can->spray(batch_size);
                current_batches++;
                all_cans_maxed = false;
            }
        }
        if (all_cans_maxed) {
            break;
        }

        // Update cans on GPU
        red_can.update_particles(time_step.count(), block_size);
        green_can.update_particles(time_step.count(), block_size);
        blue_can.update_particles(time_step.count(), block_size);

        red_can.x += -0.25;

        //std::cout << "Batches created: " << current_batches << std::endl;

        previous_frame_time = current_time;

        // Calculate elapsed time in frame
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> frame_duration = end_time - current_time;

        // Calculate remaining time needed to meet target frame rate
        std::chrono::duration<double> remaining_time = std::chrono::duration<double>(target_frame_time) - frame_duration;

        if (remaining_time.count() > 0) {
            std::this_thread::sleep_for(remaining_time);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> time_elapsed = end_time - start_time;

    std::cout << "\r\nTime to complete: " << time_elapsed.count() << std::endl;

    std::cout << "\r\nRed can:\r\nParticles created: " << red_can.particles_created << std::endl;
    //red_can.print_particles();
    std::cout << "\r\nGreen can:\r\nParticles created: " << green_can.particles_created << std::endl;
    //green_can.print_particles();
    std::cout << "\r\nBlue can:\r\nParticles created: " << blue_can.particles_created << std::endl;
    //blue_can.print_particles();
    std::cout << "\r\nTotal particles created: " << red_can.particles_created + green_can.particles_created + blue_can.particles_created << std::endl;
    std::cout << "\r\nParticles missed: " << paper.particles_missed << std::endl;
    create_bitmap(paper.paper_array, 1000, 1000, std::string("output.bmp"));

    return 0;
}