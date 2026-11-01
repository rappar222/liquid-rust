use glam::Vec3;
use std::f32::consts::PI;

/// Precomputed SPH kernel constants for high-speed evaluation
#[derive(Clone, Copy, Debug)]
pub struct SphKernels {
    pub h: f32,
    pub h2: f32,
    pub poly6_coef: f32,
    pub poly6_grad_coef: f32,
    pub poly6_lap_coef: f32,
    pub spiky_grad_coef: f32,
    pub visc_lap_coef: f32,
    pub wendland_coef: f32,
    pub wendland_grad_coef: f32,
}

impl SphKernels {
    pub fn new(h: f32) -> Self {
        let h2 = h * h;
        let h3 = h2 * h;
        let h6 = h3 * h3;
        let h9 = h6 * h3;

        let poly6_coef = 315.0 / (64.0 * PI * h9);
        let poly6_grad_coef = -945.0 / (32.0 * PI * h9);
        let poly6_lap_coef = -945.0 / (32.0 * PI * h9);
        let spiky_grad_coef = -45.0 / (PI * h6);
        let visc_lap_coef = 45.0 / (PI * h6);

        // Wendland C2 in 3D:
        let wendland_coef = 21.0 / (2.0 * PI * h3);
        let wendland_grad_coef = -210.0 / (PI * h2 * h3); // -210 / (pi * h^5)

        Self {
            h,
            h2,
            poly6_coef,
            poly6_grad_coef,
            poly6_lap_coef,
            spiky_grad_coef,
            visc_lap_coef,
            wendland_coef,
            wendland_grad_coef,
        }
    }

    /// Poly6 kernel for density estimation: W(r, h)
    #[inline(always)]
    pub fn poly6(&self, r2: f32) -> f32 {
        if r2 >= self.h2 {
            0.0
        } else {
            let diff = self.h2 - r2;
            self.poly6_coef * diff * diff * diff
        }
    }

    /// Poly6 gradient: grad W(r, h)
    #[inline(always)]
    pub fn poly6_gradient(&self, r_vec: Vec3, r2: f32) -> Vec3 {
        if r2 >= self.h2 || r2 < 1e-12 {
            Vec3::ZERO
        } else {
            let diff = self.h2 - r2;
            r_vec * (self.poly6_grad_coef * diff * diff)
        }
    }

    /// Poly6 laplacian: laplacian W(r, h)
    #[inline(always)]
    pub fn poly6_laplacian(&self, r2: f32) -> f32 {
        if r2 >= self.h2 {
            0.0
        } else {
            let diff = self.h2 - r2;
            self.poly6_lap_coef * diff * (3.0 * self.h2 - 7.0 * r2)
        }
    }

    /// Spiky gradient kernel: grad W_spiky(r, h)
    /// Prevents particle clustering at short distances
    #[inline(always)]
    pub fn spiky_gradient(&self, r_vec: Vec3, r: f32) -> Vec3 {
        if r >= self.h || r < 1e-6 {
            Vec3::ZERO
        } else {
            let diff = self.h - r;
            let factor = self.spiky_grad_coef * diff * diff / r;
            r_vec * factor
        }
    }

    /// Viscosity laplacian kernel: laplacian W_viscosity(r, h)
    #[inline(always)]
    pub fn viscosity_laplacian(&self, r: f32) -> f32 {
        if r >= self.h {
            0.0
        } else {
            self.visc_lap_coef * (self.h - r)
        }
    }

    /// Wendland C2 kernel: strictly positive with compact support and zero derivative at r=0
    #[inline(always)]
    pub fn wendland_c2(&self, r: f32) -> f32 {
        if r >= self.h {
            0.0
        } else {
            let q = r / self.h;
            let one_minus_q = 1.0 - q;
            let one_minus_q2 = one_minus_q * one_minus_q;
            let one_minus_q4 = one_minus_q2 * one_minus_q2;
            self.wendland_coef * one_minus_q4 * (1.0 + 4.0 * q)
        }
    }

    /// Wendland C2 gradient
    #[inline(always)]
    pub fn wendland_c2_gradient(&self, r_vec: Vec3, r: f32) -> Vec3 {
        if r >= self.h || r < 1e-6 {
            Vec3::ZERO
        } else {
            let diff = self.h - r;
            let factor = self.wendland_grad_coef * diff * diff * diff;
            r_vec * factor
        }
    }
}
