// LiquidRust WebGL Color Maps
// High-fidelity perceptual colormaps matching native LiquidRust

export const ColorMapType = {
    OCEAN: 'ocean',
    TURBO: 'turbo',
    VIRIDIS: 'viridis',
    FIRE: 'fire'
};

export class ColorMap {
    static sample(type, t) {
        t = Math.max(0.0, Math.min(1.0, t));
        switch (type) {
            case ColorMapType.OCEAN:
                return this.sampleOcean(t);
            case ColorMapType.TURBO:
                return this.sampleTurbo(t);
            case ColorMapType.VIRIDIS:
                return this.sampleViridis(t);
            case ColorMapType.FIRE:
                return this.sampleFire(t);
            default:
                return this.sampleOcean(t);
        }
    }

    static sampleOcean(t) {
        // Deep marine navy (0.0) -> Azure (0.35) -> Turquoise (0.7) -> White foam (1.0)
        if (t < 0.35) {
            const u = t / 0.35;
            return [
                (10 + u * (20 - 10)) / 255,
                (35 + u * (110 - 35)) / 255,
                (95 + u * (205 - 95)) / 255,
            ];
        } else if (t < 0.7) {
            const u = (t - 0.35) / 0.35;
            return [
                (20 + u * (40 - 20)) / 255,
                (110 + u * (200 - 110)) / 255,
                (205 + u * (220 - 205)) / 255,
            ];
        } else {
            const u = (t - 0.7) / 0.3;
            return [
                (40 + u * (255 - 40)) / 255,
                (200 + u * (255 - 200)) / 255,
                (220 + u * (255 - 220)) / 255,
            ];
        }
    }

    static sampleTurbo(t) {
        // Google's Turbo Colormap polynomial approximation
        const r = 0.1357 + t * (4.61539 - t * (42.6603 - t * (132.131 - t * (152.55 - t * 59.2863))));
        const g = 0.0914 + t * (2.19419 + t * (4.84297 - t * (14.185 - t * (4.27727 - t * 2.82956))));
        const b = 0.1067 + t * (12.5594 - t * (60.1972 - t * (109.075 - t * (88.508 - t * 26.8183))));
        return [
            Math.max(0, Math.min(1, r)),
            Math.max(0, Math.min(1, g)),
            Math.max(0, Math.min(1, b))
        ];
    }

    static sampleViridis(t) {
        const c0 = [0.267, 0.004, 0.329];
        const c1 = [0.190, 0.407, 0.556];
        const c2 = [0.208, 0.718, 0.472];
        const c3 = [0.993, 0.906, 0.143];

        if (t < 0.33) {
            const u = t / 0.33;
            return [c0[0] + u * (c1[0] - c0[0]), c0[1] + u * (c1[1] - c0[1]), c0[2] + u * (c1[2] - c0[2])];
        } else if (t < 0.66) {
            const u = (t - 0.33) / 0.33;
            return [c1[0] + u * (c2[0] - c1[0]), c1[1] + u * (c2[1] - c1[1]), c1[2] + u * (c2[2] - c1[2])];
        } else {
            const u = (t - 0.66) / 0.34;
            return [c2[0] + u * (c3[0] - c2[0]), c2[1] + u * (c3[1] - c2[1]), c2[2] + u * (c3[2] - c2[2])];
        }
    }

    static sampleFire(t) {
        // Black -> Dark Red -> Bright Orange -> Yellow -> White
        if (t < 0.25) {
            const u = t / 0.25;
            return [u * 0.8, 0.0, 0.0];
        } else if (t < 0.5) {
            const u = (t - 0.25) / 0.25;
            return [0.8 + u * 0.2, u * 0.5, 0.0];
        } else if (t < 0.75) {
            const u = (t - 0.5) / 0.25;
            return [1.0, 0.5 + u * 0.45, u * 0.1];
        } else {
            const u = (t - 0.75) / 0.25;
            return [1.0, 0.95 + u * 0.05, 0.1 + u * 0.9];
        }
    }
}
