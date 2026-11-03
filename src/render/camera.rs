use glam::{Mat4, Vec3};

#[derive(Clone, Debug)]
pub struct Camera {
    pub target: Vec3,
    pub azimuth: f32,   // Horizontal orbit angle in radians
    pub elevation: f32, // Vertical orbit angle in radians
    pub distance: f32,  // Distance to target
    pub fov_y: f32,     // Field of view in radians
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            target: Vec3::new(0.0, 0.8, 0.0),
            azimuth: 0.8,
            elevation: 0.35,
            distance: 3.5,
            fov_y: 45.0f32.to_radians(),
            near: 0.1,
            far: 100.0,
        }
    }
}

impl Camera {
    pub fn eye_position(&self) -> Vec3 {
        let cos_elev = self.elevation.cos();
        let sin_elev = self.elevation.sin();
        let sin_azim = self.azimuth.sin();
        let cos_azim = self.azimuth.cos();

        let offset = Vec3::new(
            self.distance * cos_elev * sin_azim,
            self.distance * sin_elev,
            self.distance * cos_elev * cos_azim,
        );

        self.target + offset
    }

    pub fn view_matrix(&self) -> Mat4 {
        let eye = self.eye_position();
        Mat4::look_at_rh(eye, self.target, Vec3::Y)
    }

    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh_gl(self.fov_y, aspect, self.near, self.far)
    }

    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        self.projection_matrix(aspect) * self.view_matrix()
    }

    pub fn rotate(&mut self, delta_azimuth: f32, delta_elevation: f32) {
        self.azimuth += delta_azimuth;
        self.elevation = (self.elevation + delta_elevation).clamp(
            -85.0f32.to_radians(),
            85.0f32.to_radians(),
        );
    }

    pub fn pan(&mut self, dx: f32, dy: f32) {
        let eye = self.eye_position();
        let forward = (self.target - eye).normalize_or_zero();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();

        let pan_speed = self.distance * 0.002;
        self.target += (-right * dx + up * dy) * pan_speed;
    }

    pub fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance * (1.0 - delta * 0.1)).clamp(0.5, 30.0);
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}
