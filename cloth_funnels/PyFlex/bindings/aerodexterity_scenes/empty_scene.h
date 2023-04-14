#pragma once
#include <iostream>
#include <vector>

class EmptyScene : public Scene
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

    EmptyScene(const char *name) : Scene(name) {}

    // void Initialize(py::array_t<float> scene_params)

    void Initialize(py::dict scene_params)
    {
        g_drawPoints = false;
        g_drawCloth = false;
        g_drawSprings = false;
        for (auto item : scene_params){
            string key = py::str(item.first);
            if (key == "radius") g_params.radius = std::stof(py::str(item.second));
            if (key == "buoyancy") g_params.buoyancy = std::stof(py::str(item.second));
            if (key == "collisionDistance") g_params.collisionDistance = std::stoi(py::str(item.second));

            if (key == "numExtraParticles") g_numExtraParticles = std::stoi(py::str(item.second));
        }

        // float stretchStiffness = 0.8;
        // float bendStiffness = 1;
        // float shearStiffness = 0.9;
        // int phase = NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter);
        // float mass = 0.5 / (64 * 64); // avg bath towel is 500-700g
        // float invMass = 1.0f / mass;
        // float velocity = 0.0f;
        // CreateSpringGrid(Vec3(0, -0.5, 0), 64, 64, 1, 0.00625, phase, stretchStiffness, bendStiffness, shearStiffness, 0.0f, 1.0f / mass);

        // int dimX = 64;
        // int dimZ = 64;
        // int baseIndex = 0;
        // float radius = 0.00625f;
        // Vec3 offset = Vec3(0.0, 0.0, 0.0);
        // Point3 p = Point3(0, 0.5f, 0);
        // Rotation r = Rotation(0.0f, 0.0f, 0.0f);
        // Mat44 transformation_mat = TransformMatrix(r, p);

        // for (int y = 0; y < dimZ; ++y){
        //     for (int x = 0; x < dimX; ++x){
        //         int index = baseIndex + y * dimX + x;
        //         Vec3 position = radius * Vec3(float(x), 0, float(y)) - offset;
        //         Point3 position_point3 = Point3(position.x, position.y, position.z);
        //         position_point3 = Multiply(transformation_mat, position_point3);

        //         // g_buffers->positions[index] = Vec4(position_point3.x, position_point3.y, position_point3.z, invMass);
        //         // g_buffers->velocities[index] = velocity;
        //         // g_buffers->phases[index] = phase;
        //         g_buffers->positions.push_back(Vec4(position_point3.x, position_point3.y, position_point3.z, invMass));
        //         g_buffers->velocities.push_back(velocity);
        //         g_buffers->phases.push_back(phase);

        //         if (x > 0 && y > 0){
        //             g_buffers->triangles.push_back(baseIndex + GridIndex(x - 1, y - 1, dimX));
        //             g_buffers->triangles.push_back(baseIndex + GridIndex(x, y - 1, dimX));
        //             g_buffers->triangles.push_back(baseIndex + GridIndex(x, y, dimX));

        //             g_buffers->triangles.push_back(baseIndex + GridIndex(x - 1, y - 1, dimX));
        //             g_buffers->triangles.push_back(baseIndex + GridIndex(x, y, dimX));
        //             g_buffers->triangles.push_back(baseIndex + GridIndex(x - 1, y, dimX));

        //             g_buffers->triangleNormals.push_back(Vec3(0.0f, 1.0f, 0.0f));
        //             g_buffers->triangleNormals.push_back(Vec3(0.0f, 1.0f, 0.0f));
        //         }
        //     }
        // }

        // // horizontal
        // for (int y = 0; y < dimZ; ++y){
        //     for (int x = 0; x < dimX; ++x){
        //         int index0 = y * dimX + x;
        //         if (x > 0){
        //             int index1 = y * dimX + x - 1;
        //             CreateSpring(baseIndex + index0, baseIndex + index1, stretchStiffness);
        //         }
        //         if (x > 1){
        //             int index2 = y * dimX + x - 2;
        //             CreateSpring(baseIndex + index0, baseIndex + index2, bendStiffness);
        //         }
        //         if (y > 0 && x < dimX - 1){
        //             int indexDiag = (y - 1) * dimX + x + 1;
        //             CreateSpring(baseIndex + index0, baseIndex + indexDiag, shearStiffness);
        //         }

        //         if (y > 0 && x > 0){
        //             int indexDiag = (y - 1) * dimX + x - 1;
        //             CreateSpring(baseIndex + index0, baseIndex + indexDiag, shearStiffness);
        //         }
        //     }
        // }

        // // vertical
        // for (int x = 0; x < dimX; ++x){
        //     for (int y = 0; y < dimZ; ++y){
        //         int index0 = y * dimX + x;
        //         if (y > 0){
        //             int index1 = (y - 1) * dimX + x;
        //             CreateSpring(baseIndex + index0, baseIndex + index1, stretchStiffness);
        //         }

        //         if (y > 1){
        //             int index2 = (y - 2) * dimX + x;
        //             CreateSpring(baseIndex + index0, baseIndex + index2, bendStiffness);
        //         }
        //     }
        // }


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
    }
};