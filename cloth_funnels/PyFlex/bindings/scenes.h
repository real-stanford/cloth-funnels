#pragma once

class Scene
{
public:
	Scene(const char *name) : mName(name) {}

	virtual void Initialize(py::dict scene_params){};
	virtual void PostInitialize() {}

	// update any buffers (all guaranteed to be mapped here)
	virtual void Update(py::array_t<float> update_params) {}

	// send any changes to flex (all buffers guaranteed to be unmapped here)
	virtual void Sync() {}

	virtual void Draw(int pass) {}
	virtual void KeyDown(int key) {}
	virtual void DoGui() {}
	virtual void CenterCamera() {}

	virtual Matrix44 GetBasis() { return Matrix44::kIdentity; }

	virtual const char *GetName() { return mName; }

	const char *mName;
};

#include "aerodexterity_scenes/empty_scene.h"
#include "softgym_scenes/softgym_cloth.h"
// #include "aerodexterity_scenes/granular_scene.h"