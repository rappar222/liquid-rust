// Orbit Camera Controller for LiquidRust Web
import { Vec3, Mat4 } from './math.js?v=20261112c';

export class Camera {
    constructor() {
        this.target = new Vec3(0.0, 0.55, 0.0);
        this.azimuth = 0.8;    // Horizontal orbit angle (radians)
        this.elevation = 0.35;  // Vertical orbit angle (radians)
        this.distance = 3.2;   // Distance to target
        this.fovY = 45.0 * (Math.PI / 180.0);
        this.near = 0.05;
        this.far = 50.0;
    }

    eyePosition() {
        const cosElev = Math.cos(this.elevation);
        const sinElev = Math.sin(this.elevation);
        const sinAzim = Math.sin(this.azimuth);
        const cosAzim = Math.cos(this.azimuth);

        const offsetX = this.distance * cosElev * sinAzim;
        const offsetY = this.distance * sinElev;
        const offsetZ = this.distance * cosElev * cosAzim;

        return new Vec3(
            this.target.x + offsetX,
            this.target.y + offsetY,
            this.target.z + offsetZ
        );
    }

    viewMatrix() {
        const eye = this.eyePosition();
        return new Mat4().lookAt(eye, this.target, new Vec3(0, 1, 0));
    }

    projectionMatrix(aspect) {
        return new Mat4().perspective(this.fovY, aspect, this.near, this.far);
    }

    rotate(deltaAzimuth, deltaElevation) {
        this.azimuth += deltaAzimuth;
        const maxElev = 85.0 * (Math.PI / 180.0);
        this.elevation = Math.max(-maxElev, Math.min(maxElev, this.elevation + deltaElevation));
    }

    pan(dx, dy) {
        const eye = this.eyePosition();
        const forward = new Vec3().subVectors(this.target, eye).normalize();
        const right = new Vec3().crossVectors(forward, new Vec3(0, 1, 0)).normalize();
        const up = new Vec3().crossVectors(right, forward).normalize();

        const panSpeed = this.distance * 0.0018;
        this.target.x += (-right.x * dx + up.x * dy) * panSpeed;
        this.target.y += (-right.y * dx + up.y * dy) * panSpeed;
        this.target.z += (-right.z * dx + up.z * dy) * panSpeed;
    }

    zoom(delta) {
        this.distance = Math.max(0.6, Math.min(25.0, this.distance * (1.0 - delta * 0.08)));
    }

    reset(is2D = false) {
        if (is2D) {
            this.target.set(0.0, 0.55, 0.0);
            this.azimuth = 0.0;
            this.elevation = 0.0;
            this.distance = 2.6;
        } else {
            this.target.set(0.0, 0.55, 0.0);
            this.azimuth = 0.8;
            this.elevation = 0.35;
            this.distance = 3.2;
        }
    }
}
