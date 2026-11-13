# LiquidRust WebGL Simulator

An interactive, GPU-accelerated WebGL 2.0 3D/2D Position-Based Fluids (PBF) simulator running directly in the browser. 

This is a standalone web version of LiquidRust designed to run in modern browsers and easily deploy anywhere with Docker Compose.

---

## 🚀 Quick Start with Docker Compose

Deploy on any server with Docker and Docker Compose installed:

```bash
docker compose up -d
```

The simulator will be immediately available at:
`http://<server-ip>:24352` (or `http://localhost:24352`)

### Custom Port
To change the host port, set the `PORT` environment variable:
```bash
PORT=3000 docker compose up -d
```

### Stop the Service
```bash
docker compose down
```

---

## 💻 Local Development (Without Docker)

You can serve the static files with any local HTTP server:

```bash
# Python 3
python3 -m http.server 8080

# Or Node.js (npx)
npx serve -l 8080 .
```

Then open `http://localhost:8080` in Chrome, Firefox, Safari, or Edge.

---

## 🎮 Controls & Interactions

### Camera Controls
| Action | Mouse | Touch |
|---|---|---|
| **3D Rotate / Orbit** | Left Click + Drag (3D Mode) | Single Finger Drag (3D Orbit) |
| **Pan / Move** | `Shift` + Drag (2D Mode) / Middle or Right Drag (3D Mode) | Two Finger Drag |
| **Zoom** | Mouse Scroll Wheel | Pinch to Zoom |
| **2D / 3D Switch** | Click **2D Mode** toggle | Tap **2D Mode** toggle |

### Fluid & Physics Interactions
| Action | Key / Gesture | Description |
|---|---|---|
| **2D Attract / Pull Fluid** | Left Click & Drag | Directly pulls particles towards the cursor in 2D mode |
| **2D Repel / Push Fluid** | Right Click (or `Ctrl` + Click) & Drag | Directly pushes particles away from the cursor in 2D mode |
| **3D Attract / Repel Fluid** | `Shift` + Left / Right Drag | Pulls or pushes particles in 3D mode |
| **Drag Sphere Obstacle** | Left Click & Drag on Sphere | Moves the dynamic collision sphere around the container |
| **ASMR Slow-Mo** | Time Speed Slider | Slows simulation down to 0.05x for ultra-smooth fluid splash visualization |

---

## 🌊 Features

- **PBF Physics Engine (Macklin & Müller 2013)**:
  - Incompressible density constraint projection with Jacobi solver iterations.
  - Artificial pressure / tensile instability correction (anti-clumping).
  - XSPH artificial viscosity for cohesive, viscous fluid motion.
  - Spatial hash grid acceleration for rapid $O(N)$ neighborhood searches.
  - Interactive collision sphere and solid wall boundaries with friction and restitution.
- **WebGL 2.0 Ray-Spherical Point Sprite Rendering**:
  - True spherical normal reconstruction (`N = (coord, sqrt(1 - r²))`) in screen space.
  - Accurate spherical depth writing (`gl_FragDepth`) ensuring **zero transparency / zero sorting artifacts** between overlapping particles.
  - Specular Blinn-Phong highlights + ambient + diffuse lighting.
  - Perceptual colormaps matching native LiquidRust: **Ocean**, **Turbo**, **Viridis**, and **Fire**.
- **Interactive UI**:
  - Presets: Dam Break, Double Dam, Splash Droplet, Sphere Obstacle, Vortex, and Zero-G.
  - Particle count adjustment (1,000 to 30,000+ particles).
  - Viscosity, surface tension, gravity vector, and sub-step controls.
  - Real-time FPS, particle count, and physics step timing counter.

---

## 🌐 Production Reverse Proxy Examples

### Nginx Reverse Proxy
```nginx
server {
    listen 80;
    server_name fluids.example.com;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Caddy
```caddy
fluids.example.com {
    reverse_proxy 127.0.0.1:8080
}
```
