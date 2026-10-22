// 3D perovskite scenes (Three.js): the animated hero lattice and the result crystal viewer.
// Exposes window.PerovskiteScene = { CrystalScene, idealSites } for app.js.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { ConvexGeometry } from "three/addons/geometries/ConvexGeometry.js";

const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const EPS = 1e-3;

// ideal cubic ABO3: A on the corners, B in the body centre, O on the face centres
export const idealSites = [
  { kind: "A", frac: [0, 0, 0] }, { kind: "B", frac: [0.5, 0.5, 0.5] },
  { kind: "O", frac: [0.5, 0.5, 0] }, { kind: "O", frac: [0.5, 0, 0.5] }, { kind: "O", frac: [0, 0.5, 0.5] },
];

function glowTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.25, "rgba(255,255,255,.55)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}

/** Periodic images of every site inside an n x n x n block (fractional, A shifted to the origin). */
function expand(sites, n, pad = 0) {
  const origin = sites[0].frac;
  const out = [];
  sites.forEach((s, i) => {
    const f = s.frac.map((c, k) => {
      const x = (((c - origin[k]) % 1) + 1) % 1;
      return x > 1 - EPS ? 0 : x;
    });
    for (let x = -pad; x <= n + pad; x++) for (let y = -pad; y <= n + pad; y++) for (let z = -pad; z <= n + pad; z++) {
      const g = [f[0] + x, f[1] + y, f[2] + z];
      const inside = g.every((c) => c >= -EPS && c <= n + EPS);
      if (pad || inside) out.push({ i, kind: s.kind, frac: g, inside });
    }
  });
  return out;
}

export class CrystalScene {
  constructor(container, { hero = false } = {}) {
    this.container = container;
    this.hero = hero;
    this.autoRotate = !reduceMotion;
    this.polyhedra = true;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(hero ? 38 : 40, 1, 0.1, 200);
    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.clock = new THREE.Clock();
    this.mouse = new THREE.Vector2();
    this.spawn = [];          // meshes animating in
    this.visible = true;

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x1b1f3b, 1.1));
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(6, 10, 8);
    this.scene.add(key);
    this.rimA = new THREE.PointLight(0x22d3ee, 60, 60);
    this.rimB = new THREE.PointLight(0xec4899, 60, 60);
    this.scene.add(this.rimA, this.rimB);

    if (hero) {
      this.scene.fog = new THREE.FogExp2(0x000000, 0.016);
      this.addParticles();
      addEventListener("pointermove", (e) => {
        this.mouse.set(e.clientX / innerWidth - 0.5, e.clientY / innerHeight - 0.5);
      }, { passive: true });
    } else {
      this.controls = new OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.enableZoom = false;   // the wheel scrolls the page
      this.controls.enablePan = false;
      this.controls.autoRotateSpeed = 1.6;
      this.controls.addEventListener("start", () => (this.userBusy = true));
      this.controls.addEventListener("end", () => (this.userBusy = false));
    }

    new ResizeObserver(() => this.resize()).observe(container);
    new IntersectionObserver(([e]) => (this.visible = e.isIntersecting)).observe(container);
    this.resize();
    this.renderer.setAnimationLoop(() => this.tick());
  }

  addParticles() {
    const n = 380, pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
    const palette = [new THREE.Color("#22d3ee"), new THREE.Color("#a78bfa"), new THREE.Color("#f472b6")];
    for (let i = 0; i < n; i++) {
      const r = 9 + Math.random() * 22, t = Math.random() * Math.PI * 2, p = Math.acos(2 * Math.random() - 1);
      pos.set([r * Math.sin(p) * Math.cos(t), r * Math.sin(p) * Math.sin(t), r * Math.cos(p)], i * 3);
      const c = palette[i % 3];
      col.set([c.r, c.g, c.b], i * 3);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    this.particles = new THREE.Points(geo, new THREE.PointsMaterial({
      size: 0.35, map: glowTexture(), vertexColors: true, transparent: true, opacity: 0.8,
      depthWrite: false, blending: THREE.AdditiveBlending,
    }));
    this.scene.add(this.particles);
  }

  colors() {
    return { A: new THREE.Color(css("--site-a")), B: new THREE.Color(css("--site-b")), O: new THREE.Color(css("--site-o")) };
  }

  /** sites: [{kind:'A'|'B'|'O', frac:[x,y,z]}], a: lattice constant (Angstrom), n: cells per edge. */
  setStructure({ sites, a, n = 1, bond = null }) {
    this.state = { sites, a, n, bond };
    this.build(true);
  }

  build(animate = false) {
    if (!this.state) return;
    const { sites, a, n } = this.state;
    this.root.clear();
    this.spawn = [];
    const col = this.colors();
    const light = document.documentElement.dataset.theme === "light" ||
      (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: light)").matches);
    const half = (n * a) / 2;
    const toVec = (f) => new THREE.Vector3(f[0] * a - half, f[1] * a - half, f[2] * a - half);

    const all = expand(sites, n, 1);
    const shown = all.filter((t) => t.inside);
    const oxygens = all.filter((t) => t.kind === "O").map((t) => toVec(t.frac));
    const bond = this.state.bond || a / 2;
    const radius = { A: 0.62, B: 0.4, O: 0.32 };
    const mats = {};
    for (const k of ["A", "B", "O"]) {
      mats[k] = new THREE.MeshPhysicalMaterial({
        color: col[k], metalness: 0.15, roughness: 0.22, clearcoat: 1, clearcoatRoughness: 0.15,
        emissive: col[k], emissiveIntensity: this.hero ? 0.35 : light ? 0.05 : 0.18,
      });
    }
    const sphere = new THREE.SphereGeometry(1, 40, 28);
    const center = new THREE.Vector3();

    for (const t of shown) {
      const m = new THREE.Mesh(sphere, mats[t.kind]);
      m.position.copy(toVec(t.frac));
      m.userData.r = radius[t.kind];
      m.scale.setScalar(animate && !reduceMotion ? 0.001 : m.userData.r);
      m.userData.delay = m.position.distanceTo(center) * 0.06;
      this.root.add(m);
      this.spawn.push(m);
    }

    // BO6 octahedra: the six nearest O around every B (images outside the block included)
    const polyMat = new THREE.MeshPhysicalMaterial({
      color: col.B, transparent: true, opacity: this.hero ? 0.16 : light ? 0.28 : 0.22, roughness: 0.1,
      metalness: 0, side: THREE.DoubleSide, depthWrite: false, emissive: col.B, emissiveIntensity: this.hero ? 0.4 : 0.15,
    });
    const edgeMat = new THREE.LineBasicMaterial({ color: col.B, transparent: true, opacity: this.hero ? 0.55 : 0.75 });
    const bondMat = new THREE.MeshStandardMaterial({ color: col.B, roughness: 0.35, metalness: 0.1, emissive: col.B, emissiveIntensity: 0.15 });
    const cyl = new THREE.CylinderGeometry(0.07, 0.07, 1, 12);
    for (const b of shown.filter((t) => t.kind === "B")) {
      const p = toVec(b.frac);
      const verts = oxygens.map((o) => [o, o.distanceTo(p)]).sort((x, y) => x[1] - y[1]).slice(0, 6)
        .filter(([, d]) => d < bond * 1.6).map(([o]) => o);
      if (verts.length < 4) continue;
      const group = new THREE.Group();
      if (this.polyhedra) {
        const geo = new ConvexGeometry(verts);
        group.add(new THREE.Mesh(geo, polyMat));
        group.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), edgeMat));
      }
      if (!this.hero) {
        for (const v of verts) {
          const m = new THREE.Mesh(cyl, bondMat);
          const mid = p.clone().add(v).multiplyScalar(0.5);
          m.position.copy(mid);
          m.scale.y = p.distanceTo(v);
          m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), v.clone().sub(p).normalize());
          group.add(m);
        }
      }
      group.userData.delay = p.distanceTo(center) * 0.06 + 0.25;
      group.userData.r = 1;
      if (animate && !reduceMotion) group.scale.setScalar(0.001);
      this.root.add(group);
      this.spawn.push(group);
    }

    // unit-cell edges
    const L = n * a, h = L / 2, pts = [];
    const c = [[-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h], [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]];
    [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
      .forEach(([i, j]) => pts.push(...c[i], ...c[j]));
    const cellGeo = new THREE.BufferGeometry();
    cellGeo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    const cell = new THREE.LineSegments(cellGeo, new THREE.LineDashedMaterial({
      color: new THREE.Color(css("--muted")), dashSize: 0.25, gapSize: 0.18, transparent: true, opacity: this.hero ? 0.25 : 0.7,
    }));
    cell.computeLineDistances();
    this.root.add(cell);

    this.rimA.color.set(css("--site-b"));
    this.rimB.color.set(css("--accent-3"));
    if (this.scene.fog) this.scene.fog.color.set(css("--bg"));
    this.frame(L);
    this.spawnStart = this.clock.getElapsedTime();
  }

  frame(L) {
    const dist = L * (this.hero ? 2.5 : 2.25) + (this.hero ? 4 : 3.2);
    this.camera.position.set(dist * 0.62, dist * 0.42, dist * 0.68);
    this.camera.lookAt(0, 0, 0);
    this.rimA.position.set(L, L, -L);
    this.rimB.position.set(-L, -L * 0.5, L);
    if (this.controls) { this.controls.target.set(0, 0, 0); this.controls.update(); }
  }

  setOptions({ n, polyhedra, autoRotate } = {}) {
    if (autoRotate !== undefined) this.autoRotate = autoRotate;
    let rebuild = false;
    if (polyhedra !== undefined && polyhedra !== this.polyhedra) { this.polyhedra = polyhedra; rebuild = true; }
    if (n !== undefined && this.state && n !== this.state.n) { this.state.n = n; rebuild = true; }
    if (rebuild) this.build(true);
  }

  refreshTheme() { this.build(false); }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  tick() {
    if (!this.visible) return;
    const t = this.clock.getElapsedTime();
    // atoms grow in from the centre outwards with a slight overshoot
    if (this.spawn.length && this.spawnStart !== undefined) {
      let done = true;
      for (const m of this.spawn) {
        const k = Math.min(1, Math.max(0, (t - this.spawnStart - m.userData.delay) / 0.55));
        if (k < 1) done = false;
        const ease = 1 + 2.2 * Math.pow(k - 1, 3) + 1.2 * Math.pow(k - 1, 2);   // easeOutBack
        m.scale.setScalar(Math.max(0.001, m.userData.r * (k === 0 ? 0 : ease)));
      }
      if (done) this.spawn = [];
    }
    if (this.hero) {
      if (!reduceMotion) {
        this.root.rotation.y = t * 0.12;
        this.root.rotation.x = Math.sin(t * 0.18) * 0.18;
        this.root.position.y = Math.sin(t * 0.6) * 0.15;
        if (this.particles) this.particles.rotation.y = -t * 0.02;
      }
      this.camera.position.x += (this.mouse.x * 3 + this.camPos().x - this.camera.position.x) * 0.04;
      this.camera.position.y += (-this.mouse.y * 2 + this.camPos().y - this.camera.position.y) * 0.04;
      this.camera.lookAt(0, 0, 0);
    } else if (this.controls) {
      this.controls.autoRotate = this.autoRotate && !this.userBusy;
      this.controls.update();
    }
    this.renderer.render(this.scene, this.camera);
  }

  camPos() {
    if (!this._cam) {
      const L = this.state.n * this.state.a, dist = L * 2.5 + 4;
      this._cam = new THREE.Vector3(dist * 0.62, dist * 0.42, dist * 0.68);
    }
    return this._cam;
  }
}

window.PerovskiteScene = { CrystalScene, idealSites };
window.dispatchEvent(new Event("scene-ready"));
