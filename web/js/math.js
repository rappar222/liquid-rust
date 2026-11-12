// 3D Math Library for LiquidRust WebGL
// High-performance vector, matrix, and geometric calculations

export class Vec3 {
    constructor(x = 0, y = 0, z = 0) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    set(x, y, z) {
        this.x = x;
        this.y = y;
        this.z = z;
        return this;
    }

    clone() {
        return new Vec3(this.x, this.y, this.z);
    }

    copy(v) {
        this.x = v.x;
        this.y = v.y;
        this.z = v.z;
        return this;
    }

    add(v) {
        this.x += v.x;
        this.y += v.y;
        this.z += v.z;
        return this;
    }

    addVectors(a, b) {
        this.x = a.x + b.x;
        this.y = a.y + b.y;
        this.z = a.z + b.z;
        return this;
    }

    sub(v) {
        this.x -= v.x;
        this.y -= v.y;
        this.z -= v.z;
        return this;
    }

    subVectors(a, b) {
        this.x = a.x - b.x;
        this.y = a.y - b.y;
        this.z = a.z - b.z;
        return this;
    }

    scale(s) {
        this.x *= s;
        this.y *= s;
        this.z *= s;
        return this;
    }

    dot(v) {
        return this.x * v.x + this.y * v.y + this.z * v.z;
    }

    cross(v) {
        const x = this.y * v.z - this.z * v.y;
        const y = this.z * v.x - this.x * v.z;
        const z = this.x * v.y - this.y * v.x;
        this.x = x;
        this.y = y;
        this.z = z;
        return this;
    }

    lengthSq() {
        return this.x * this.x + this.y * this.y + this.z * this.z;
    }

    length() {
        return Math.sqrt(this.lengthSq());
    }

    normalize() {
        const len = this.length();
        if (len > 1e-6) {
            this.scale(1.0 / len);
        } else {
            this.set(0, 0, 0);
        }
        return this;
    }

    distanceTo(v) {
        const dx = this.x - v.x;
        const dy = this.y - v.y;
        const dz = this.z - v.z;
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    distanceToSq(v) {
        const dx = this.x - v.x;
        const dy = this.y - v.y;
        const dz = this.z - v.z;
        return dx * dx + dy * dy + dz * dz;
    }
}

export class Mat4 {
    constructor() {
        this.elements = new Float32Array(16);
        this.identity();
    }

    copy(m) {
        this.elements.set(m.elements);
        return this;
    }

    clone() {
        const m = new Mat4();
        m.copy(this);
        return m;
    }

    identity() {
        const te = this.elements;
        te[0] = 1; te[4] = 0; te[8] = 0;  te[12] = 0;
        te[1] = 0; te[5] = 1; te[9] = 0;  te[13] = 0;
        te[2] = 0; te[6] = 0; te[10] = 1; te[14] = 0;
        te[3] = 0; te[7] = 0; te[11] = 0; te[15] = 1;
        return this;
    }

    perspective(fovRad, aspect, near, far) {
        const te = this.elements;
        const f = 1.0 / Math.tan(fovRad / 2.0);
        const nf = 1.0 / (near - far);

        te[0] = f / aspect;
        te[1] = 0;
        te[2] = 0;
        te[3] = 0;

        te[4] = 0;
        te[5] = f;
        te[6] = 0;
        te[7] = 0;

        te[8] = 0;
        te[9] = 0;
        te[10] = (far + near) * nf;
        te[11] = -1.0;

        te[12] = 0;
        te[13] = 0;
        te[14] = 2.0 * far * near * nf;
        te[15] = 0;

        return this;
    }

    lookAt(eye, target, up) {
        const te = this.elements;

        const z0 = eye.x - target.x;
        const z1 = eye.y - target.y;
        const z2 = eye.z - target.z;
        let len = 1.0 / Math.hypot(z0, z1, z2);
        const zx = z0 * len, zy = z1 * len, zz = z2 * len;

        const x0 = up.y * zz - up.z * zy;
        const x1 = up.z * zx - up.x * zz;
        const x2 = up.x * zy - up.y * zx;
        len = 1.0 / Math.hypot(x0, x1, x2);
        const xx = x0 * len, xy = x1 * len, xz = x2 * len;

        const yx = zy * xz - zz * xy;
        const yy = zz * xx - zx * xz;
        const yz = zx * xy - zy * xx;

        te[0] = xx;  te[1] = yx;  te[2] = zx;  te[3] = 0;
        te[4] = xy;  te[5] = yy;  te[6] = zy;  te[7] = 0;
        te[8] = xz;  te[9] = yz;  te[10] = zz; te[11] = 0;
        te[12] = -(xx * eye.x + xy * eye.y + xz * eye.z);
        te[13] = -(yx * eye.x + yy * eye.y + yz * eye.z);
        te[14] = -(zx * eye.x + zy * eye.y + zz * eye.z);
        te[15] = 1;

        return this;
    }

    multiply(m) {
        return this.multiplyMatrices(this, m);
    }

    multiplyMatrices(a, b) {
        const ae = a.elements;
        const be = b.elements;
        const te = this.elements;

        const a11 = ae[0], a12 = ae[4], a13 = ae[8],  a14 = ae[12];
        const a21 = ae[1], a22 = ae[5], a23 = ae[9],  a24 = ae[13];
        const a31 = ae[2], a32 = ae[6], a33 = ae[10], a34 = ae[14];
        const a41 = ae[3], a42 = ae[7], a43 = ae[11], a44 = ae[15];

        const b11 = be[0], b12 = be[4], b13 = be[8],  b14 = be[12];
        const b21 = be[1], b22 = be[5], b23 = be[9],  b24 = be[13];
        const b31 = be[2], b32 = be[6], b33 = be[10], b34 = be[14];
        const b41 = be[3], b42 = be[7], b43 = be[11], b44 = be[15];

        te[0] = a11 * b11 + a12 * b21 + a13 * b31 + a14 * b41;
        te[4] = a11 * b12 + a12 * b22 + a13 * b32 + a14 * b42;
        te[8] = a11 * b13 + a12 * b23 + a13 * b33 + a14 * b43;
        te[12] = a11 * b14 + a12 * b24 + a13 * b34 + a14 * b44;

        te[1] = a21 * b11 + a22 * b21 + a23 * b31 + a24 * b41;
        te[5] = a21 * b12 + a22 * b22 + a23 * b32 + a24 * b42;
        te[9] = a21 * b13 + a22 * b23 + a23 * b33 + a24 * b43;
        te[13] = a21 * b14 + a22 * b24 + a23 * b34 + a24 * b44;

        te[2] = a31 * b11 + a32 * b21 + a33 * b31 + a34 * b41;
        te[6] = a31 * b12 + a32 * b22 + a33 * b32 + a34 * b42;
        te[10] = a31 * b13 + a32 * b23 + a33 * b33 + a34 * b43;
        te[14] = a31 * b14 + a32 * b24 + a33 * b34 + a34 * b44;

        te[3] = a41 * b11 + a42 * b21 + a43 * b31 + a44 * b41;
        te[7] = a41 * b12 + a42 * b22 + a43 * b32 + a44 * b42;
        te[11] = a41 * b13 + a42 * b23 + a43 * b33 + a44 * b43;
        te[15] = a41 * b14 + a42 * b24 + a43 * b34 + a44 * b44;

        return this;
    }

    invert() {
        const te = this.elements;
        const n11 = te[0], n21 = te[1], n31 = te[2], n41 = te[3];
        const n12 = te[4], n22 = te[5], n32 = te[6], n42 = te[7];
        const n13 = te[8], n23 = te[9], n33 = te[10], n43 = te[11];
        const n14 = te[12], n24 = te[13], n34 = te[14], n44 = te[15];

        const t11 = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44;
        const t12 = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44 - n12 * n33 * n44;
        const t13 = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44 + n12 * n23 * n44;
        const t14 = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;

        const det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
        if (det === 0) return this.identity();

        const invDet = 1.0 / det;

        te[0] = t11 * invDet;
        te[1] = (n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 + n23 * n31 * n44 - n21 * n33 * n44) * invDet;
        te[2] = (n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44 + n21 * n32 * n44) * invDet;
        te[3] = (n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43 - n21 * n32 * n43) * invDet;

        te[4] = t12 * invDet;
        te[5] = (n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44 + n11 * n33 * n44) * invDet;
        te[6] = (n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44 - n11 * n32 * n44) * invDet;
        te[7] = (n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43 + n11 * n32 * n43) * invDet;

        te[8] = t13 * invDet;
        te[9] = (n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44 - n11 * n23 * n44) * invDet;
        te[10] = (n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 - n12 * n21 * n44 + n11 * n22 * n44) * invDet;
        te[11] = (n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 + n12 * n21 * n43 - n11 * n22 * n43) * invDet;

        te[12] = t14 * invDet;
        te[13] = (n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34 + n11 * n23 * n34) * invDet;
        te[14] = (n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34 - n11 * n22 * n34) * invDet;
        te[15] = (n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33 + n11 * n22 * n33) * invDet;

        return this;
    }
}

export class Aabb {
    constructor(min = new Vec3(-1, 0, -0.5), max = new Vec3(1, 1.2, 0.5)) {
        this.min = min;
        this.max = max;
    }

    size() {
        return new Vec3(this.max.x - this.min.x, this.max.y - this.min.y, this.max.z - this.min.z);
    }

    center() {
        return new Vec3(
            (this.min.x + this.max.x) * 0.5,
            (this.min.y + this.max.y) * 0.5,
            (this.min.z + this.max.z) * 0.5
        );
    }
}

export class Ray {
    constructor(origin = new Vec3(), direction = new Vec3(0, 0, -1)) {
        this.origin = origin;
        this.direction = direction.normalize();
    }

    distanceToPoint(point) {
        const v = new Vec3().subVectors(point, this.origin);
        const proj = v.dot(this.direction);
        if (proj < 0) {
            return v.length();
        }
        const perp = new Vec3().copy(this.direction).scale(proj).sub(v);
        return perp.length();
    }
}
