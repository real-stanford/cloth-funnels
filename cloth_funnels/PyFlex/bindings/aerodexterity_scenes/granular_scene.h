#pragma once
#include <iostream>
#include <vector>

class GranularScene : public Scene
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

    GranularScene(const char *name) : Scene(name) {}

    // void Initialize(py::array_t<float> scene_params)

    void Initialize(py::array_t<float> scene_params)
    {
        // auto ptr = (float *)scene_params.request().ptr;
        // int render_type = ptr[0]; // 0: only points, 1: only mesh, 2: points + mesh
        // g_drawPoints = render_type & 1;
        // g_drawCloth = (render_type & 2) >> 1;
        // g_drawSprings = false;

        // cam_x = ptr[1];
        // cam_y = ptr[2];
        // cam_z = ptr[3];
        // cam_angle_x = ptr[4];
        // cam_angle_y = ptr[5];
        // cam_angle_z = ptr[6];
        // cam_width = int(ptr[7]);
        // cam_height = int(ptr[8]);


        // granular pile
		float radius = 0.075f;

		Vec3 lower(8.0f, 4.0f, 2.0f);

		CreateParticleShape(GetFilePathByPlatform("../../data/sphere.ply").c_str(), lower, 1.0f, 0.0f, radius, 0.0f, 0.f, true, 1.0f, NvFlexMakePhase(1, 0), true, 0.00f);
		g_numSolidParticles = g_buffers->positions.size();

		CreateParticleShape(GetFilePathByPlatform("../../data/sandcastle.obj").c_str(), Vec3(-2.0f, -radius * 0.15f, 0.0f), 4.0f, 0.0f, radius * 1.0001f, 0.0f, 1.0f, false, 0.0f, NvFlexMakePhase(0, eNvFlexPhaseSelfCollide), false, 0.00f);

		g_numSubsteps = 2;

		g_params.radius = radius;
		g_params.staticFriction = 1.0f;
		g_params.dynamicFriction = 0.5f;
		g_params.viscosity = 0.0f;
		g_params.numIterations = 12;
		g_params.particleCollisionMargin = g_params.radius * 0.25f; // 5% collision margin
		g_params.sleepThreshold = g_params.radius * 0.25f;
		g_params.shockPropagation = 6.f;
		g_params.restitution = 0.2f;
		g_params.relaxationFactor = 1.f;
		g_params.damping = 0.14f;
		g_params.numPlanes = 1;

		// draw options
		g_drawPoints = true;
		g_warmup = false;

		// hack, change the color of phase 0 particles to 'sand'
		g_colors[0] = Colour(0.805f, 0.702f, 0.401f);

        cout<<"\nHERE\n"<<endl;
        cout<<GetFilePathByPlatform("../../data/sphere.ply").c_str()<<endl;

        
        MapBuffers(g_buffers);
        cout<<g_buffers->activeIndices.size()<<endl;
        cout<<g_buffers->positions.size()<<endl;
        cout<<g_buffers->velocities.size()<<endl;
        UnmapBuffers(g_buffers);

    }

    virtual void CenterCamera(void)
    {
        g_camPos = Vec3(cam_x, cam_y, cam_z);
        g_camAngle = Vec3(cam_angle_x, cam_angle_y, cam_angle_z);
        g_screenHeight = cam_height;
        g_screenWidth = cam_width;
    }

    // void Update()
	// {
	// 	// launch ball after 3 seconds
	// 	if (g_frame == 180)
	// 	{
	// 		for (int i = 0; i < g_numSolidParticles; ++i)
	// 		{
	// 			g_buffers->positions[i].w = 0.9f;
	// 			g_buffers->velocities[i] = Vec3(-15.0f, 0.0f, 0.0f);
	// 		}
	// 	}
	// }
};