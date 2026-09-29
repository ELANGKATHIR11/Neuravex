/*
 * Neuravex Embedded C Header (libneuravex).
 *
 * Minimal, portable C/C++ interface for running lightweight Neuravex variants
 * (pico / femto) on edge and microcontroller targets:
 *   - ESP32-S3 (ESP-NN / ESP-DL)
 *   - Arduino / Teensy / Cortex-M
 *   - Raspberry Pi Zero / Cortex-A
 *   - Standard POSIX / Windows C runtimes
 */

#ifndef LIBNEURAVEX_H
#define LIBNEURAVEX_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

#define NEURAVEX_MAX_OBJECTS 64
#define NEURAVEX_MAX_CLASSES 80

typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
} nx_bbox2d_t;

typedef struct {
    float x;
    float y;
    float z;
    float l;
    float w;
    float h;
    float yaw;
} nx_bbox3d_t;

typedef struct {
    int32_t id;
    int32_t class_id;
    float score;
    nx_bbox2d_t bbox2d;
    nx_bbox3d_t bbox3d;
    float depth;
    float distance;
} nx_object_t;

typedef struct {
    uint32_t frame_id;
    uint32_t width;
    uint32_t height;
    uint32_t num_objects;
    nx_object_t objects[NEURAVEX_MAX_OBJECTS];
} nx_frame_state_t;

typedef struct {
    float fx;
    float fy;
    float cx;
    float cy;
} nx_camera_intrinsics_t;

/* Context handle for embedded engine instance */
typedef struct nx_context_s* nx_context_t;

/**
 * @brief Initialize a Neuravex embedded perception engine.
 * @param model_weights Pointer to flat buffer or TFLite/Micro model binary.
 * @param weights_size Size of model weights buffer in bytes.
 * @param intrinsics Camera pinhole intrinsics.
 * @return Context pointer, or NULL on failure.
 */
nx_context_t nx_init(const uint8_t* model_weights, size_t weights_size, const nx_camera_intrinsics_t* intrinsics);

/**
 * @brief Run inference on RGB888 interleaved frame buffer.
 * @param ctx Valid context pointer.
 * @param rgb888_data Pointer to image buffer (size = width * height * 3).
 * @param width Image width in pixels.
 * @param height Image height in pixels.
 * @param conf_threshold Minimum confidence threshold [0.0, 1.0].
 * @param out_state Pre-allocated output frame state to populate.
 * @return 0 on success, negative error code on failure.
 */
int nx_process_frame(
    nx_context_t ctx,
    const uint8_t* rgb888_data,
    uint32_t width,
    uint32_t height,
    float conf_threshold,
    nx_frame_state_t* out_state
);

/**
 * @brief Destroy context and release all allocated resources.
 */
void nx_destroy(nx_context_t ctx);

#ifdef __cplusplus
}
#endif

#endif /* LIBNEURAVEX_H */
