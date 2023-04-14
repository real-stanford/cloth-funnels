#pragma once
#include <iostream>
#include <vector>

inline void swap(int &a, int &b)
{
    int tmp = a;
    a = b;
    b = tmp;
}

class SoftgymCloth : public Scene
{
public:
    float cam_x;
    float cam_y;
    float cam_z;
    float cam_angle_x;
    float cam_angle_y;
    float cam_angle_z;
    int cam_width;
    int cam_height;

    SoftgymCloth(const char *name) : Scene(name) {}

    float get_param_float(py::array_t<float> scene_params, int idx)
    {
        auto ptr = (float *)scene_params.request().ptr;
        float out = ptr[idx];
        return out;
    }

    void Initialize(py::dict scene_params)
    {
        float initX = 0;
        float initY = 0.5;
        float initZ = 0;

        int dimx = 64;
        int dimz = 64;
        float radius = 0.00625f;

        int render_type = 2; // 0: only points, 1: only mesh, 2: points + mesh

        cam_x = 0;
        cam_y = 1.2;
        cam_z = 1.1;
        cam_angle_x = 0;
        cam_angle_y = -3.14159/4;
        cam_angle_z = 0;
        cam_width = 720;
        cam_height = 720;

        float stretchStiffness = 0.8;
        float bendStiffness = 1;
        float shearStiffness = 0.9;
        int phase = NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter);

        int flip_mesh = 0; // Flip half

        // auto ptr = (float *)scene_params.request().ptr;
        // float initX = ptr[0];
        // float initY = ptr[1];
        // float initZ = ptr[2];

        // int dimx = (int)ptr[3];
        // int dimz = (int)ptr[4];
        // float radius = 0.00625f;

        // int render_type = ptr[8]; // 0: only points, 1: only mesh, 2: points + mesh

        // cam_x = ptr[9];
        // cam_y = ptr[10];
        // cam_z = ptr[11];
        // cam_angle_x = ptr[12];
        // cam_angle_y = ptr[13];
        // cam_angle_z = ptr[14];
        // cam_width = int(ptr[15]);
        // cam_height = int(ptr[16]);

        // float stretchStiffness = ptr[5];
        // float bendStiffness = ptr[6];
        // float shearStiffness = ptr[7];
        // int phase = NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter);

        // int flip_mesh = int(ptr[18]); // Flip half

        {
            float mass = 0.5 / (dimx * dimz); // avg bath towel is 500-700g
            CreateSpringGrid(Vec3(initX, -initY, initZ), dimx, dimz, 1, radius, phase, stretchStiffness, bendStiffness, shearStiffness, 0.0f, 1.0f / mass);
        }
        // Flip the last half of the mesh for the folding task
        if (flip_mesh)
        {
            int size = g_buffers->triangles.size();
            for (int j = 0; j < int((dimz - 1)); ++j)
                for (int i = int((dimx - 1) * 1 / 8); i < int((dimx - 1) * 1 / 8) + 5; ++i)
                {
                    int idx = j * (dimx - 1) + i;

                    if ((i != int((dimx - 1) * 1 / 8 + 4)))
                        swap(g_buffers->triangles[idx * 3 * 2], g_buffers->triangles[idx * 3 * 2 + 1]);
                    if ((i != int((dimx - 1) * 1 / 8)))
                        swap(g_buffers->triangles[idx * 3 * 2 + 3], g_buffers->triangles[idx * 3 * 2 + 4]);
                }
        }

        g_numSubsteps = 4;
        g_params.numIterations = 30;

        g_params.dynamicFriction = 0.75f;
        g_params.particleFriction = 1.0f;
        g_params.damping = 1.0f;
        g_params.sleepThreshold = 0.02f;

        g_params.relaxationFactor = 1.0f;
        g_params.shapeCollisionMargin = 0.04f;

        g_sceneLower = Vec3(-1.0f);
        g_sceneUpper = Vec3(1.0f);
        g_drawPoints = false;

        g_params.radius = radius * 1.8f;
        g_params.collisionDistance = 0.005f;

        g_drawPoints = render_type & 1;
        g_drawCloth = (render_type & 2) >> 1;
        g_drawSprings = false;
    }

    virtual void CenterCamera(void)
    {
        g_camPos = Vec3(cam_x, cam_y, cam_z);
        g_camAngle = Vec3(cam_angle_x, cam_angle_y, cam_angle_z);
        g_screenHeight = cam_height;
        g_screenWidth = cam_width;
    }
};