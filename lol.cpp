#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>
#include <SDL2/SDL.h>
#include <optional>

struct Vec3f {
    float x, y, z;
};

Vec3f make_vec3f(float x, float y, float z) {
    return {x, y, z};
}

float vec3f_dot(const Vec3f &a, const Vec3f &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3f vec3f_add(const Vec3f &a, const Vec3f &b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3f vec3f_sub(const Vec3f &a, const Vec3f &b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3f vec3f_scale(const Vec3f &v, float scalar) {
    return {v.x * scalar, v.y * scalar, v.z * scalar};
}

float vec3f_norm(const Vec3f &v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

Vec3f vec3f_normalize(const Vec3f &v) {
    float n = vec3f_norm(v);
    return n > 0 ? make_vec3f(v.x / n, v.y / n, v.z / n) : make_vec3f(0, 0, 0);
}

const float sphere_radius = 1.5;
const float noise_amplitude = 1.0;

template <typename T> 
T lerp(const T &v0, const T &v1, float t) {
    return v0 + (v1 - v0) * std::max(0.f, std::min(1.f, t));
}

float hash(float n) {
    return sin(n) * 43758.5453f - floor(sin(n) * 43758.5453f);
}

float noise(const Vec3f &x) {
    Vec3f p = {floor(x.x), floor(x.y), floor(x.z)};
    Vec3f f = {x.x - p.x, x.y - p.y, x.z - p.z};
    f = vec3f_scale(f, vec3f_dot(f, vec3f_sub({3.f, 3.f, 3.f}, vec3f_scale(f, 2.f))));
    float n = vec3f_dot(p, {1.f, 57.f, 113.f});
    return lerp(
                lerp(lerp(hash(n + 0.f), hash(n + 1.f), f.x), lerp(hash(n + 57.f), hash(n + 58.f), f.x), f.y),
                lerp(lerp(hash(n + 113.f), hash(n + 114.f), f.x), lerp(hash(n + 170.f), hash(n + 171.f), f.x), f.y),
                f.z);
}

Vec3f rotate(const Vec3f &v) {
    return {
        vec3f_dot({0.00, 0.80, 0.60}, v), 
        vec3f_dot({-0.80, 0.36, -0.48}, v), 
        vec3f_dot({-0.60, -0.48, 0.64}, v)
    };
}

float fractal_brownian_motion(const Vec3f &x) {
    Vec3f p = rotate(x);
    float f = 0;
    f += 0.5000 * noise(p); p = vec3f_scale(p, 2.32);
    f += 0.2500 * noise(p); p = vec3f_scale(p, 3.03);
    f += 0.1250 * noise(p); p = vec3f_scale(p, 2.61);
    f += 0.0625 * noise(p);
    return f / 0.9375;
}

float signed_distance(const Vec3f &p) {
    float displacement = -fractal_brownian_motion(vec3f_scale(p, 3.4)) * noise_amplitude;
    return vec3f_norm(p) - (sphere_radius + displacement);
}

Vec3f distance_field_normal(const Vec3f &pos) {
    const float eps = 0.1;
    float d = signed_distance(pos);
    float nx = signed_distance(vec3f_add(pos, {eps, 0, 0})) - d;
    float ny = signed_distance(vec3f_add(pos, {0, eps, 0})) - d;
    float nz = signed_distance(vec3f_add(pos, {0, 0, eps})) - d;
    return vec3f_normalize({nx, ny, nz});
}

std::optional<Vec3f> sphere_trace(const Vec3f &orig, const Vec3f &dir) {
    Vec3f pos = orig;
    for (size_t i = 0; i < 128; i++) {
        float d = signed_distance(pos);
        if (d < 0) return pos;
        pos = vec3f_add(pos, vec3f_scale(dir, std::max(d * 0.1f, 0.01f)));
    }
    return std::nullopt;
}

std::vector<Vec3f> render_framebuffer(int width, int height, float fov) {
    std::vector<Vec3f> framebuffer(width * height);

    #pragma omp parallel for
    for (size_t j = 0; j < height; j++) {
        for (size_t i = 0; i < width; i++) {
            float dir_x = (i + 0.5f) - width / 2.0f;
            float dir_y = -(j + 0.5f) + height / 2.0f;
            float dir_z = -height / (2.0f * tan(fov / 2.0f));
            Vec3f dir = vec3f_normalize({dir_x, dir_y, dir_z});
            auto hit = sphere_trace({0, 0, 3}, dir);
            if (hit) {
                Vec3f light_dir = vec3f_normalize(vec3f_sub({0, 10, 10}, *hit));
                float light_intensity = std::max(0.4f, vec3f_dot(light_dir, distance_field_normal(*hit)));
                framebuffer[i + j * width] = vec3f_scale({1, 1, 1}, light_intensity);
            } else {
                framebuffer[i + j * width] = {0.3f, 0.9f, 0.2f};
            }
        }
    }

    return framebuffer;
}

std::vector<uint8_t> convert_framebuffer_to_pixels(const std::vector<Vec3f> &framebuffer, int width, int height) {
    std::vector<uint8_t> pixels(width * height * 3);

    for (size_t j = 0; j < height; j++) {
        for (size_t i = 0; i < width; i++) {
            const Vec3f &color = framebuffer[i + j * width];
            uint8_t r = static_cast<uint8_t>(std::max(0.f, std::min(255.f, color.x * 255.f)));
            uint8_t g = static_cast<uint8_t>(std::max(0.f, std::min(255.f, color.y * 255.f)));
            uint8_t b = static_cast<uint8_t>(std::max(0.f, std::min(255.f, color.z * 255.f)));

            size_t pixelIndex = (j * width + i) * 3;
            pixels[pixelIndex] = r;
            pixels[pixelIndex + 1] = g;
            pixels[pixelIndex + 2] = b;
        }
    }

    return pixels;
}

int main() {
    const int width = 640;
    const int height = 480;
    const float fov = M_PI / 3.0;

    auto framebuffer = render_framebuffer(width, height, fov);
    auto pixels = convert_framebuffer_to_pixels(framebuffer, width, height);

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cerr << "Error initializing SDL: " << SDL_GetError() << std::endl;
        return 1;
    }

    SDL_Window *window = SDL_CreateWindow("Sphere Trace", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, SDL_WINDOW_SHOWN);
    if (!window) {
        std::cerr << "Error creating SDL window: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return 1;
    }

    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        std::cerr << "Error creating SDL renderer: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, width, height);
    if (!texture) {
        std::cerr << "Error creating SDL texture: " << SDL_GetError() << std::endl;
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    bool running = true;
    SDL_Event event;

    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }
        }

        SDL_UpdateTexture(texture, nullptr, pixels.data(), width * 3);
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);
    }

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
