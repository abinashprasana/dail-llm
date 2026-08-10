import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useLayoutEffect, useMemo, useRef } from "react";
import type { MutableRefObject, RefObject } from "react";
import {
  BoxGeometry,
  BufferGeometry,
  CanvasTexture,
  Color,
  DoubleSide,
  DynamicDrawUsage,
  Float32BufferAttribute,
  LineBasicMaterial,
  MathUtils,
  MeshStandardMaterial,
  Object3D,
  PlaneGeometry,
  QuadraticBezierCurve3,
  SRGBColorSpace,
  Vector3,
} from "three";
import type { Group, InstancedMesh } from "three";
import type { Material } from "three";
import { TRACE_STEPS } from "./heroTrace";
import type { TraceMotion, TraceStage } from "./heroTrace";

const GLYPHS = ["D", "Á", "I", "L", "É", "R", "E", "N"];
const GLYPH_TEXTURE_SIZE = 128;
const SHADOW_TEXTURE_SIZE = 128;
const ARC_START = MathUtils.degToRad(-25);
const ARC_END = MathUtils.degToRad(205);
const AISLE_ANGLE = Math.PI / 2;
const ASSEMBLY_SECONDS = 1.15;
const ACTIVE_TILE_INDEX = 2;
const PREDICTED_TILE_INDEX = 6;
const BASE_YAW = -0.055;
const MAX_POINTER_YAW = MathUtils.degToRad(1.45);
const MAX_POINTER_PITCH = MathUtils.degToRad(1.05);

type Tuple3 = [number, number, number];

interface Segment {
  angle: number;
  position: Tuple3;
  rotationY: number;
  width: number;
}

interface Tier {
  radius: number;
  y: number;
  segments: Segment[];
}

interface Transform {
  position: Tuple3;
  rotationX?: number;
  rotationY: number;
  rotationZ?: number;
  scale: Tuple3;
}

interface CharacterTile extends Segment {
  char: string;
  y: number;
}

interface PointerTarget {
  x: number;
  y: number;
}

interface SceneGeometry {
  box: BoxGeometry;
  plane: PlaneGeometry;
}

export interface ChamberCanvasProps {
  active: boolean;
  traceStage: TraceStage;
  traceRunId: number;
  traceMotion: TraceMotion;
  onReady?: () => void;
  onAssembled?: () => void;
  onError?: (error?: unknown) => void;
}

function createTier(radius: number, y: number, count = 17): Tier {
  const step = (ARC_END - ARC_START) / (count - 1);
  const width = radius * step * 0.94;
  const segments = Array.from({ length: count }, (_, index) => {
    const angle = ARC_START + step * index;
    return {
      angle,
      position: [Math.cos(angle) * radius, y, Math.sin(angle) * radius] as Tuple3,
      rotationY: angle + Math.PI / 2,
      width,
    };
  }).filter((segment) => Math.abs(segment.angle - AISLE_ANGLE) > step * 0.48);

  return { radius, y, segments };
}

function offsetPosition(segment: Segment, radial: number, y: number): Tuple3 {
  return [
    segment.position[0] + Math.cos(segment.angle) * radial,
    y,
    segment.position[2] + Math.sin(segment.angle) * radial,
  ];
}

const TIERS = [createTier(2.28, 0.01), createTier(3.02, 0.31), createTier(3.76, 0.61)];

const TILE_SPECS: Array<[number, number]> = [
  [0, 1], [0, 4], [0, 9], [0, 13],
  [1, 2], [1, 11], [2, 4], [2, 10],
];

const CHARACTER_TILES: CharacterTile[] = TILE_SPECS.map(([tierIndex, segmentIndex], index) => {
  const tier = TIERS[tierIndex];
  const segment = tier.segments[segmentIndex];
  return {
    ...segment,
    position: offsetPosition(segment, -0.19, tier.y + 0.415),
    y: tier.y + 0.415,
    char: GLYPHS[index],
  };
});

const sourceTile = CHARACTER_TILES[ACTIVE_TILE_INDEX];
const predictedTile = CHARACTER_TILES[PREDICTED_TILE_INDEX];
const sourcePoint = new Vector3(sourceTile.position[0], sourceTile.y + 0.07, sourceTile.position[2]);
const predictedPoint = new Vector3(predictedTile.position[0], predictedTile.y + 0.07, predictedTile.position[2]);

const ATTENTION_CURVES = Array.from({ length: 8 }, (_, index) => {
  const midpoint = sourcePoint.clone().lerp(predictedPoint, 0.5);
  const spread = index - 3.5;
  const control = midpoint.add(new Vector3(spread * 0.11, 0.68 + index * 0.035, Math.sin(index * 1.7) * 0.16));
  return new QuadraticBezierCurve3(sourcePoint.clone(), control, predictedPoint.clone());
});

const TRACE_SECONDS = TRACE_STEPS.reduce<Record<TraceStage, number>>(
  (durations, step) => ({ ...durations, [step.stage]: step.durationMs / 1000 }),
  { idle: 0, speaker: 0, attention: 0, prediction: 0 },
);

const PREDICTION_GREEN = new Color("#3d735c");
const PREDICTION_BRASS = new Color("#c9a55c");
const PREDICTION_GREEN_EMISSIVE = new Color("#70ae8f");
const PREDICTION_BRASS_EMISSIVE = new Color("#d8bc77");

function createAttentionGeometry(): BufferGeometry {
  const positions: number[] = [];
  const colors: number[] = [];
  ATTENTION_CURVES.forEach((curve, curveIndex) => {
    const color = new Color(curveIndex === 2 || curveIndex === 6 ? "#c9a55c" : "#9aafa5");
    const points = curve.getPoints(24);
    for (let index = 0; index < points.length - 1; index += 1) {
      positions.push(...points[index].toArray(), ...points[index + 1].toArray());
      colors.push(color.r, color.g, color.b, color.r, color.g, color.b);
    }
  });
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new Float32BufferAttribute(colors, 3));
  geometry.computeBoundingSphere();
  return geometry;
}

function createGlyphTexture(char: string): CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = GLYPH_TEXTURE_SIZE;
  canvas.height = GLYPH_TEXTURE_SIZE;
  const context = canvas.getContext("2d");
  if (context) {
    context.font = `600 ${GLYPH_TEXTURE_SIZE * 0.65}px "Newsreader", Georgia, serif`;
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillStyle = "#ffffff";
    context.fillText(char, GLYPH_TEXTURE_SIZE / 2, GLYPH_TEXTURE_SIZE * 0.53);
  }
  const texture = new CanvasTexture(canvas);
  texture.colorSpace = SRGBColorSpace;
  return texture;
}

function createShadowTexture(): CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = SHADOW_TEXTURE_SIZE;
  canvas.height = SHADOW_TEXTURE_SIZE;
  const context = canvas.getContext("2d");
  if (context) {
    const gradient = context.createRadialGradient(64, 64, 8, 64, 64, 62);
    gradient.addColorStop(0, "rgba(255,255,255,0.72)");
    gradient.addColorStop(0.58, "rgba(255,255,255,0.28)");
    gradient.addColorStop(1, "rgba(255,255,255,0)");
    context.fillStyle = gradient;
    context.fillRect(0, 0, SHADOW_TEXTURE_SIZE, SHADOW_TEXTURE_SIZE);
  }
  const texture = new CanvasTexture(canvas);
  texture.colorSpace = SRGBColorSpace;
  return texture;
}

function useInstanceMatrices(ref: RefObject<InstancedMesh | null>, transforms: Transform[], dynamic = false) {
  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    const dummy = new Object3D();

    transforms.forEach((transform, index) => {
      dummy.position.set(...transform.position);
      dummy.rotation.set(transform.rotationX ?? 0, transform.rotationY, transform.rotationZ ?? 0);
      dummy.scale.set(...transform.scale);
      dummy.updateMatrix();
      mesh.setMatrixAt(index, dummy.matrix);
    });

    if (dynamic) mesh.instanceMatrix.setUsage(DynamicDrawUsage);
    mesh.instanceMatrix.needsUpdate = true;
    mesh.computeBoundingSphere();
  }, [dynamic, ref, transforms]);
}

function BenchTier({ tier, tone, geometry }: { tier: Tier; tone: string; geometry: SceneGeometry }) {
  const steps = useRef<InstancedMesh>(null);
  const rails = useRef<InstancedMesh>(null);
  const cushions = useRef<InstancedMesh>(null);
  const backs = useRef<InstancedMesh>(null);

  const stepTransforms = useMemo<Transform[]>(
    () => tier.segments.map((segment) => ({
      position: offsetPosition(segment, 0, tier.y - 0.09),
      rotationY: segment.rotationY,
      scale: [segment.width, 0.25, 0.93],
    })),
    [tier],
  );
  const railTransforms = useMemo<Transform[]>(
    () => tier.segments.map((segment) => ({
      position: offsetPosition(segment, -0.2, tier.y + 0.205),
      rotationY: segment.rotationY,
      scale: [segment.width * 0.99, 0.25, 0.26],
    })),
    [tier],
  );
  const cushionTransforms = useMemo<Transform[]>(
    () => tier.segments.map((segment) => ({
      position: offsetPosition(segment, 0.12, tier.y + 0.315),
      rotationY: segment.rotationY,
      scale: [segment.width * 0.77, 0.075, 0.4],
    })),
    [tier],
  );
  const backTransforms = useMemo<Transform[]>(
    () => tier.segments.map((segment) => ({
      position: offsetPosition(segment, 0.34, tier.y + 0.54),
      rotationY: segment.rotationY,
      scale: [segment.width * 0.75, 0.38, 0.1],
    })),
    [tier],
  );

  useInstanceMatrices(steps, stepTransforms);
  useInstanceMatrices(rails, railTransforms);
  useInstanceMatrices(cushions, cushionTransforms);
  useInstanceMatrices(backs, backTransforms);

  return (
    <>
      <instancedMesh ref={steps} args={[undefined, undefined, tier.segments.length]}>
        <primitive object={geometry.box} attach="geometry" />
        <meshStandardMaterial color="#102019" roughness={0.91} metalness={0.01} />
      </instancedMesh>
      <instancedMesh ref={rails} args={[undefined, undefined, tier.segments.length]}>
        <primitive object={geometry.box} attach="geometry" />
        <meshStandardMaterial color="#332c21" roughness={0.65} metalness={0.05} />
      </instancedMesh>
      <instancedMesh ref={cushions} args={[undefined, undefined, tier.segments.length]}>
        <primitive object={geometry.box} attach="geometry" />
        <meshStandardMaterial color={tone} roughness={0.88} metalness={0.01} />
      </instancedMesh>
      <instancedMesh ref={backs} args={[undefined, undefined, tier.segments.length]}>
        <primitive object={geometry.box} attach="geometry" />
        <meshStandardMaterial color={tone} roughness={0.82} metalness={0.01} />
      </instancedMesh>
    </>
  );
}

function LayerInlays({ geometry }: { geometry: SceneGeometry }) {
  const inlays = useRef<InstancedMesh>(null);
  const transforms = useMemo<Transform[]>(() => {
    const items: Transform[] = [];
    for (let band = 0; band < 4; band += 1) {
      const radius = 1.0 + band * 0.23;
      const count = 19;
      const step = (ARC_END - ARC_START) / (count - 1);
      for (let index = 0; index < count; index += 1) {
        const angle = ARC_START + step * index;
        if (Math.abs(angle - AISLE_ANGLE) < step * 0.48) continue;
        items.push({
          position: [Math.cos(angle) * radius, -0.164 + band * 0.002, Math.sin(angle) * radius],
          rotationY: angle + Math.PI / 2,
          scale: [radius * step * 0.94, 0.014, 0.068],
        });
      }
    }
    return items;
  }, []);

  useInstanceMatrices(inlays, transforms);

  return (
    <instancedMesh ref={inlays} args={[undefined, undefined, transforms.length]}>
      <primitive object={geometry.box} attach="geometry" />
      <meshStandardMaterial color="#18352b" roughness={0.96} metalness={0} />
    </instancedMesh>
  );
}

function CharacterField({ textures, geometry }: { textures: Map<string, CanvasTexture>; geometry: SceneGeometry }) {
  const bases = useRef<InstancedMesh>(null);
  const transforms = useMemo<Transform[]>(
    () => CHARACTER_TILES.map((tile) => ({
      position: tile.position,
      rotationY: tile.rotationY,
      scale: [0.34, 0.025, 0.2],
    })),
    [],
  );

  useInstanceMatrices(bases, transforms);

  return (
    <>
      <instancedMesh ref={bases} args={[undefined, undefined, transforms.length]}>
        <primitive object={geometry.box} attach="geometry" />
        <meshStandardMaterial color="#42594f" roughness={0.66} metalness={0.12} />
      </instancedMesh>
      {CHARACTER_TILES.map((tile, index) => index === ACTIVE_TILE_INDEX || index === PREDICTED_TILE_INDEX ? null : (
        <group key={`${tile.char}-${index}`} position={[tile.position[0], tile.y + 0.017, tile.position[2]]} rotation={[0, tile.rotationY, 0]}>
          <mesh rotation={[-Math.PI / 2, 0, 0]} scale={[0.17, 0.17, 1]}>
            <primitive object={geometry.plane} attach="geometry" />
            <meshBasicMaterial
              map={textures.get(tile.char)}
              color="#d8d4c2"
              transparent
              alphaTest={0.08}
              side={DoubleSide}
              depthWrite={false}
            />
          </mesh>
        </group>
      ))}
    </>
  );
}

function GroundStage({ shadowTexture, geometry }: { shadowTexture: CanvasTexture; geometry: SceneGeometry }) {
  return (
    <>
      <mesh position={[0.18, -0.53, 0.22]} rotation={[-Math.PI / 2, 0, 0]} scale={[5.35, 4.15, 1]}>
        <primitive object={geometry.plane} attach="geometry" />
        <meshBasicMaterial map={shadowTexture} color="#000000" transparent opacity={0.48} depthWrite={false} />
      </mesh>
      <mesh position={[0, -0.39, 0.03]} scale={[1.08, 1, 0.8]}>
        <cylinderGeometry args={[4.16, 4.24, 0.2, 80]} />
        <meshStandardMaterial color="#050c09" roughness={0.95} metalness={0.01} />
      </mesh>
      <mesh position={[0, -0.265, 0.03]} scale={[1.06, 1, 0.79]}>
        <cylinderGeometry args={[4.02, 4.08, 0.08, 80]} />
        <meshStandardMaterial color="#153127" roughness={0.9} metalness={0.01} />
      </mesh>
      <mesh position={[0, -0.205, 0.12]} scale={[1.03, 1, 0.8]}>
        <cylinderGeometry args={[2.15, 2.2, 0.045, 64]} />
        <meshStandardMaterial color="#0c211a" roughness={0.94} metalness={0} />
      </mesh>
      <mesh position={[0, -0.16, 0.34]} scale={[0.58, 0.025, 4.4]}>
        <primitive object={geometry.box} attach="geometry" />
        <meshStandardMaterial color="#15392d" roughness={0.9} />
      </mesh>
      <mesh position={[0, -0.142, 0.34]} scale={[0.055, 0.012, 4.42]}>
        <primitive object={geometry.box} attach="geometry" />
        <meshBasicMaterial color="#9a7a3d" transparent opacity={0.3} />
      </mesh>
      <LayerInlays geometry={geometry} />
    </>
  );
}

function Dais({
  geometry,
  brassMaterial,
}: {
  geometry: SceneGeometry;
  brassMaterial: RefObject<MeshStandardMaterial | null>;
}) {
  const timber = useRef<InstancedMesh>(null);
  const upholstery = useRef<InstancedMesh>(null);
  const brass = useRef<InstancedMesh>(null);
  const timberTransforms = useMemo<Transform[]>(() => [
    { position: [0, -0.1, -2.68], rotationY: 0, scale: [3.15, 0.2, 1.18] },
    { position: [0, 0.68, -3.03], rotationY: 0, scale: [2.84, 1.35, 0.18] },
    { position: [0, 0.38, -2.6], rotationY: 0, scale: [2.55, 0.64, 0.62] },
  ], []);
  const upholsteryTransforms = useMemo<Transform[]>(() => [
    { position: [0, 0.72, -2.93], rotationY: 0, scale: [2.15, 0.82, 0.045] },
    { position: [0, 0.73, -2.58], rotationY: 0, scale: [2.72, 0.09, 0.74] },
  ], []);
  const brassTransforms = useMemo<Transform[]>(() => [
    { position: [0, 0.76, -2.2], rotationY: 0, scale: [2.56, 0.045, 0.035] },
    { position: [0, 0.61, -1.42], rotationX: -0.16, rotationY: 0, scale: [0.68, 0.09, 0.46] },
  ], []);

  useInstanceMatrices(timber, timberTransforms);
  useInstanceMatrices(upholstery, upholsteryTransforms);
  useInstanceMatrices(brass, brassTransforms);

  return (
    <>
      <instancedMesh ref={timber} args={[undefined, undefined, timberTransforms.length]}>
        <primitive object={geometry.box} attach="geometry" />
        <meshStandardMaterial color="#332c21" roughness={0.67} metalness={0.05} />
      </instancedMesh>
      <instancedMesh ref={upholstery} args={[undefined, undefined, upholsteryTransforms.length]}>
        <primitive object={geometry.box} attach="geometry" />
        <meshStandardMaterial color="#1b563f" roughness={0.79} />
      </instancedMesh>
      <instancedMesh ref={brass} args={[undefined, undefined, brassTransforms.length]}>
        <primitive object={geometry.box} attach="geometry" />
        <meshStandardMaterial
          ref={brassMaterial}
          color="#b99349"
          emissive="#8f6d31"
          emissiveIntensity={0.04}
          roughness={0.33}
          metalness={0.61}
        />
      </instancedMesh>
      <mesh position={[0, 0.22, -1.42]}>
        <cylinderGeometry args={[0.085, 0.115, 0.74, 12]} />
        <meshStandardMaterial color="#a27e3d" roughness={0.34} metalness={0.68} />
      </mesh>
    </>
  );
}

interface RevealMaterialState {
  opacity: number;
  transparent: boolean;
  depthWrite: boolean;
}

function revealMaterial(material: Material, progress: number) {
  const stateKey = "chamberReveal";
  const userData = material.userData as Record<string, unknown>;
  if (!userData[stateKey]) {
    userData[stateKey] = {
      opacity: material.opacity,
      transparent: material.transparent,
      depthWrite: material.depthWrite,
    } satisfies RevealMaterialState;
  }
  const initial = userData[stateKey] as RevealMaterialState;
  material.opacity = initial.opacity * progress;
  const fading = progress < 0.999;
  const transparent = initial.transparent || fading;
  if (material.transparent !== transparent) {
    material.transparent = transparent;
    material.needsUpdate = true;
  }
  material.depthWrite = initial.depthWrite && !fading;
}

function setReveal(group: Group | null, progress: number, rise: number) {
  if (!group) return;
  const eased = 1 - Math.pow(1 - MathUtils.clamp(progress, 0, 1), 3);
  group.visible = eased > 0.001;
  group.position.y = (eased - 1) * rise;
  group.scale.setScalar(0.985 + eased * 0.015);
  group.traverse((node) => {
    const material = (node as unknown as { material?: Material | Material[] }).material;
    if (Array.isArray(material)) material.forEach((entry) => revealMaterial(entry, eased));
    else if (material) revealMaterial(material, eased);
  });
}

function revealAt(elapsed: number, start: number, end: number) {
  return MathUtils.clamp((elapsed - start) / (end - start), 0, 1);
}

function hideBeads(mesh: InstancedMesh | null, dummy: Object3D) {
  if (!mesh) return;
  ATTENTION_CURVES.forEach((_, index) => {
    dummy.position.copy(sourcePoint);
    dummy.scale.setScalar(0.001);
    dummy.updateMatrix();
    mesh.setMatrixAt(index, dummy.matrix);
  });
  mesh.instanceMatrix.needsUpdate = true;
}

function Chamber({
  active,
  traceStage,
  traceRunId,
  traceMotion,
  pointerTarget,
  onAssembled,
}: Pick<ChamberCanvasProps, "active" | "traceStage" | "traceRunId" | "traceMotion" | "onAssembled"> & {
  pointerTarget: MutableRefObject<PointerTarget>;
}) {
  const invalidate = useThree((state) => state.invalidate);
  const gl = useThree((state) => state.gl);
  const chamber = useRef<Group>(null);
  const ground = useRef<Group>(null);
  const tierGroups = useRef<Array<Group | null>>([]);
  const dais = useRef<Group>(null);
  const tiles = useRef<Group>(null);
  const attention = useRef<Group>(null);
  const beads = useRef<InstancedMesh>(null);
  const speakerPlate = useRef<Group>(null);
  const predictedPlate = useRef<Group>(null);
  const speakerMaterial = useRef<MeshStandardMaterial>(null);
  const predictedMaterial = useRef<MeshStandardMaterial>(null);
  const lecternMaterial = useRef<MeshStandardMaterial>(null);
  const pathMaterial = useRef<LineBasicMaterial>(null);
  const assemblyTime = useRef(0);
  const assembledNotified = useRef(false);
  const traceElapsed = useRef(0);
  const traceAnimating = useRef(false);
  const beadDummy = useMemo(() => new Object3D(), []);
  const geometry = useMemo<SceneGeometry>(() => ({ box: new BoxGeometry(1, 1, 1), plane: new PlaneGeometry(1, 1) }), []);
  const attentionGeometry = useMemo(createAttentionGeometry, []);

  const glyphTextures = useMemo(() => {
    const map = new Map<string, CanvasTexture>();
    new Set(GLYPHS).forEach((char) => map.set(char, createGlyphTexture(char)));
    return map;
  }, []);
  const shadowTexture = useMemo(createShadowTexture, []);

  useEffect(() => () => {
    glyphTextures.forEach((texture) => texture.dispose());
    shadowTexture.dispose();
    geometry.box.dispose();
    geometry.plane.dispose();
    attentionGeometry.dispose();
  }, [attentionGeometry, geometry, glyphTextures, shadowTexture]);

  useLayoutEffect(() => {
    const mesh = beads.current;
    if (!mesh) return;
    mesh.instanceMatrix.setUsage(DynamicDrawUsage);
    hideBeads(mesh, beadDummy);
  }, [beadDummy]);

  useEffect(() => {
    traceElapsed.current = traceMotion === "animate" ? 0 : TRACE_SECONDS[traceStage];
    traceAnimating.current = traceMotion === "animate" && TRACE_SECONDS[traceStage] > 0;
    invalidate();
  }, [invalidate, traceMotion, traceRunId, traceStage]);

  useEffect(() => {
    if (active) invalidate();
  }, [active, invalidate]);

  useEffect(() => {
    const canvas = gl.domElement;
    const recenter = () => {
      pointerTarget.current.x = 0;
      pointerTarget.current.y = 0;
      invalidate();
    };
    const onPointerMove = (event: PointerEvent) => {
      if (event.pointerType !== "mouse" && event.pointerType !== "pen") return;
      const bounds = canvas.getBoundingClientRect();
      if (!bounds.width || !bounds.height) return;
      pointerTarget.current.x = MathUtils.clamp(((event.clientX - bounds.left) / bounds.width) * 2 - 1, -1, 1);
      pointerTarget.current.y = MathUtils.clamp(-(((event.clientY - bounds.top) / bounds.height) * 2 - 1), -1, 1);
      invalidate();
    };
    const onPointerEnd = (event: PointerEvent) => {
      if (event.pointerType === "mouse" || event.pointerType === "pen") recenter();
    };

    canvas.addEventListener("pointermove", onPointerMove, { passive: true });
    canvas.addEventListener("pointerleave", onPointerEnd, { passive: true });
    canvas.addEventListener("pointercancel", onPointerEnd, { passive: true });
    canvas.addEventListener("pointerup", onPointerEnd, { passive: true });
    window.addEventListener("blur", recenter);
    return () => {
      canvas.removeEventListener("pointermove", onPointerMove);
      canvas.removeEventListener("pointerleave", onPointerEnd);
      canvas.removeEventListener("pointercancel", onPointerEnd);
      canvas.removeEventListener("pointerup", onPointerEnd);
      window.removeEventListener("blur", recenter);
    };
  }, [gl, invalidate, pointerTarget]);

  useFrame((_, delta) => {
    if (!active) return;

    if (assemblyTime.current < ASSEMBLY_SECONDS) {
      assemblyTime.current = Math.min(ASSEMBLY_SECONDS, assemblyTime.current + delta);
      const t = assemblyTime.current;
      setReveal(ground.current, revealAt(t, 0, 0.3), 0.08);
      setReveal(dais.current, revealAt(t, 0.16, 0.52), 0.13);
      setReveal(tierGroups.current[0], revealAt(t, 0.27, 0.65), 0.16);
      setReveal(tierGroups.current[1], revealAt(t, 0.4, 0.8), 0.18);
      setReveal(tierGroups.current[2], revealAt(t, 0.53, 0.95), 0.2);
      setReveal(tiles.current, revealAt(t, 0.75, 1.08), 0.1);
      setReveal(attention.current, revealAt(t, 0.92, ASSEMBLY_SECONDS), 0.05);
      invalidate();
      if (assemblyTime.current >= ASSEMBLY_SECONDS && !assembledNotified.current) {
        assembledNotified.current = true;
        onAssembled?.();
      }
    }

    const root = chamber.current;
    if (root) {
      const targetYaw = BASE_YAW + pointerTarget.current.x * MAX_POINTER_YAW;
      const targetPitch = pointerTarget.current.y * MAX_POINTER_PITCH;
      root.rotation.y = MathUtils.damp(root.rotation.y, targetYaw, 7, delta);
      root.rotation.x = MathUtils.damp(root.rotation.x, targetPitch, 7, delta);
      if (Math.abs(root.rotation.y - targetYaw) > 0.0002 || Math.abs(root.rotation.x - targetPitch) > 0.0002) {
        invalidate();
      }
    }

    if (traceAnimating.current) {
      traceElapsed.current = Math.min(TRACE_SECONDS[traceStage], traceElapsed.current + delta);
    }
    const duration = TRACE_SECONDS[traceStage];
    const progress = duration > 0 ? MathUtils.clamp(traceElapsed.current / duration, 0, 1) : 1;
    const eased = 1 - Math.pow(1 - progress, 3);

    let speakerGlow = 0.04;
    let predictedGlow = 0.04;
    let pathOpacity = 0;
    let speakerLift = 0;
    let predictedLift = 0;
    if (traceStage === "speaker") {
      speakerGlow = 0.08 + eased * 0.92;
      speakerLift = eased * 0.065;
    } else if (traceStage === "attention") {
      speakerGlow = 0.66;
      speakerLift = 0.065;
      pathOpacity = 0.035 + eased * 0.205;
      predictedGlow = 0.06 + eased * 0.12;
    } else if (traceStage === "prediction") {
      speakerGlow = 0.58 - eased * 0.32;
      speakerLift = 0.065 * (1 - eased);
      predictedLift = eased * 0.035;
      pathOpacity = 0.16 * (1 - eased) + 0.025;
      predictedGlow = 0.1 + eased * 1.05;
    }

    if (speakerMaterial.current) speakerMaterial.current.emissiveIntensity = speakerGlow;
    if (predictedMaterial.current) {
      predictedMaterial.current.emissiveIntensity = predictedGlow;
      const resolution = traceStage === "prediction" ? eased : 0;
      predictedMaterial.current.color.lerpColors(PREDICTION_GREEN, PREDICTION_BRASS, resolution);
      predictedMaterial.current.emissive.lerpColors(PREDICTION_GREEN_EMISSIVE, PREDICTION_BRASS_EMISSIVE, resolution);
    }
    if (speakerPlate.current) speakerPlate.current.position.y = sourceTile.y + 0.004 + speakerLift;
    if (predictedPlate.current) predictedPlate.current.position.y = predictedTile.y + 0.004 + predictedLift;
    if (lecternMaterial.current) lecternMaterial.current.emissiveIntensity = 0.04 + speakerGlow * 0.12;
    if (pathMaterial.current) pathMaterial.current.opacity = pathOpacity;

    const beadMesh = beads.current;
    if (traceStage === "attention" && traceMotion === "animate" && traceAnimating.current && beadMesh) {
      ATTENTION_CURVES.forEach((curve, index) => {
        const local = MathUtils.clamp(progress * 1.16 - index * 0.023, 0, 1);
        beadDummy.position.copy(curve.getPoint(local));
        beadDummy.scale.setScalar(Math.max(0.001, Math.sin(Math.PI * local) * 0.9));
        beadDummy.updateMatrix();
        beadMesh.setMatrixAt(index, beadDummy.matrix);
      });
      beadMesh.instanceMatrix.needsUpdate = true;
    } else {
      hideBeads(beadMesh, beadDummy);
    }

    if (traceAnimating.current && traceElapsed.current < duration) {
      invalidate();
    } else {
      traceAnimating.current = false;
    }
  });

  return (
    <group ref={chamber} position={[0, 0.28, 0]} rotation={[0, BASE_YAW, 0]}>
      <group ref={ground}><GroundStage shadowTexture={shadowTexture} geometry={geometry} /></group>

      {TIERS.map((tier, index) => (
        <group key={tier.radius} ref={(node) => { tierGroups.current[index] = node; }}>
          <BenchTier tier={tier} tone={index === 1 ? "#1d654a" : "#18543f"} geometry={geometry} />
        </group>
      ))}

      <group ref={dais}>
        <Dais geometry={geometry} brassMaterial={lecternMaterial} />
      </group>

      <group ref={tiles}>
        <CharacterField textures={glyphTextures} geometry={geometry} />
        <group ref={speakerPlate} position={[sourceTile.position[0], sourceTile.y + 0.004, sourceTile.position[2]]} rotation={[0, sourceTile.rotationY, 0]}>
          <mesh scale={[0.39, 0.028, 0.24]}>
            <primitive object={geometry.box} attach="geometry" />
            <meshStandardMaterial ref={speakerMaterial} color="#8b713b" emissive="#d5b564" emissiveIntensity={0.04} roughness={0.45} metalness={0.18} />
          </mesh>
          <mesh position={[0, 0.019, 0]} rotation={[-Math.PI / 2, 0, 0]} scale={[0.17, 0.17, 1]}>
            <primitive object={geometry.plane} attach="geometry" />
            <meshBasicMaterial map={glyphTextures.get(sourceTile.char)} color="#f3ead4" transparent alphaTest={0.08} side={DoubleSide} depthWrite={false} />
          </mesh>
        </group>
        <group ref={predictedPlate} position={[predictedTile.position[0], predictedTile.y + 0.004, predictedTile.position[2]]} rotation={[0, predictedTile.rotationY, 0]}>
          <mesh scale={[0.39, 0.028, 0.24]}>
            <primitive object={geometry.box} attach="geometry" />
            <meshStandardMaterial ref={predictedMaterial} color="#3d735c" emissive="#70ae8f" emissiveIntensity={0.04} roughness={0.48} metalness={0.12} />
          </mesh>
          <mesh position={[0, 0.019, 0]} rotation={[-Math.PI / 2, 0, 0]} scale={[0.17, 0.17, 1]}>
            <primitive object={geometry.plane} attach="geometry" />
            <meshBasicMaterial map={glyphTextures.get(predictedTile.char)} color="#f3ead4" transparent alphaTest={0.08} side={DoubleSide} depthWrite={false} />
          </mesh>
        </group>
      </group>

      <group ref={attention}>
        <lineSegments geometry={attentionGeometry}>
          <lineBasicMaterial ref={pathMaterial} vertexColors transparent opacity={0} depthWrite={false} />
        </lineSegments>
        <instancedMesh ref={beads} args={[undefined, undefined, ATTENTION_CURVES.length]} frustumCulled={false}>
          <sphereGeometry args={[0.038, 8, 8]} />
          <meshBasicMaterial color="#d8bc77" transparent opacity={0.86} depthWrite={false} />
        </instancedMesh>
      </group>
    </group>
  );
}

function CameraAim() {
  const camera = useThree((state) => state.camera);
  useLayoutEffect(() => {
    camera.lookAt(0, -0.22, 0.08);
    camera.updateProjectionMatrix();
  }, [camera]);
  return null;
}

export default function ChamberCanvas({
  active,
  traceStage,
  traceRunId,
  traceMotion,
  onReady,
  onAssembled,
  onError,
}: ChamberCanvasProps) {
  const pointerTarget = useRef<PointerTarget>({ x: 0, y: 0 });

  return (
    <Canvas
      className="chamber-canvas"
      aria-hidden="true"
      tabIndex={-1}
      dpr={[1, 1.5]}
      camera={{ position: [3.15, 4.35, 11], fov: 44 }}
      frameloop="demand"
      gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
      onCreated={({ gl }) => {
        gl.setClearAlpha(0);
        gl.domElement.addEventListener("webglcontextlost", (event) => {
          event.preventDefault();
          onError?.(new Error("WebGL context lost"));
        }, { once: true });
        onReady?.();
      }}
    >
      <CameraAim />
      <hemisphereLight args={["#d7d1bd", "#06100c", 1.3]} />
      <directionalLight position={[4.8, 6.5, 5.5]} intensity={2.75} color="#f0dfba" />
      <directionalLight position={[-4.5, 3.2, 1.5]} intensity={1.15} color="#4b9f7d" />
      <directionalLight position={[0, 4.2, -5]} intensity={1.4} color="#c9a55c" />
      <pointLight position={[0, 2.2, -2.25]} intensity={6} distance={6.5} color="#c9a55c" />
      <Chamber
        active={active}
        traceStage={traceStage}
        traceRunId={traceRunId}
        traceMotion={traceMotion}
        pointerTarget={pointerTarget}
        onAssembled={onAssembled}
      />
    </Canvas>
  );
}
