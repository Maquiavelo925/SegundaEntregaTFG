# ================================================================
#  MakiMalum — Generador procedural de ciudades medievales
#  Autor   : Makiavelo925, Álex Cuevas Langa
#  Versión : 1.2.0  |  Blender 4.2+
# ================================================================

bl_info = {
    "name":        "MakiMalum",
    "author":      "Makiavelo925, Álex Cuevas Langa",
    "version":     (1, 5, 0),
    "blender":     (4, 2, 0),
    "location":    "View3D › N-Panel › MakiMalum",
    "description": "Genera ciudades medievales procedurales sobre un plano subdividido",
    "category":    "Add Mesh",
    "doc_url":     "",
    "tracker_url": "",
}

import bpy
import bmesh
import random
from collections import deque
from mathutils import Matrix
from bpy.props import (
    BoolProperty,
    EnumProperty, IntProperty, FloatProperty, PointerProperty,
)
from bpy.types import PropertyGroup


# ================================================================
#  CONSTANTES — niveles de ciudad
# ================================================================

# Orden de generación: los niveles más bajos tienen prioridad
LEVELS = ["plaza", "iglesia", "orden1", "orden2", "orden3", "orden4"]

LEVEL_LABELS = {
    "plaza":   "Plaza Central",
    "iglesia": "Iglesia",
    "orden1":  "Edificios 1er orden",
    "orden2":  "Edificios 2do orden",
    "orden3":  "Edificios 3er orden",
    "orden4":  "Edificios 4to orden",
}

LEVEL_ICONS = {
    "plaza":   "WORLD",
    "iglesia": "GHOST_ENABLED",
    "orden1":  "SOLO_ON",
    "orden2":  "SOLO_OFF",
    "orden3":  "DOT",
    "orden4":  "RADIOBUT_OFF",
}

# Distancia BFS → nivel asignado (para los niveles automáticos)
# La iglesia se trata aparte (1 por P, cara adyacente aleatoria)
DIST_TO_LEVEL = {
    0: "plaza",
    1: "orden1",   # orden1 cubre el anillo 1 salvo la cara de iglesia
    2: "orden2",
    3: "orden3",
}
# dist >= 4 → "orden4"


# ================================================================
#  UTILIDADES — colecciones
# ================================================================

def collection_items(self, context):
    items = [("NONE", "— Ninguna —", "")]
    for col in bpy.data.collections:
        items.append((col.name, col.name, ""))
    return items


def get_collection(col_or_name):
    """Devuelve la colección. Acepta objeto Collection o nombre string."""
    if col_or_name is None:
        return None
    if isinstance(col_or_name, str):
        if not col_or_name or col_or_name == "NONE":
            return None
        return bpy.data.collections.get(col_or_name)
    return col_or_name   # ya es un objeto Collection


def random_obj_from_collection(col, rng):
    if col is None:
        return None
    objs = [o for o in col.objects if o.type == 'MESH']
    return rng.choice(objs) if objs else None


# ================================================================
#  UTILIDADES — geometría
# ================================================================

def face_center_world(face, obj_matrix):
    """Centro de la cara en espacio mundo."""
    return obj_matrix @ face.calc_center_median()


def face_normal_world(face, obj_matrix):
    return (obj_matrix.to_3x3() @ face.normal).normalized()


def place_obj_on_face(context, source_obj, center_w, normal_w,
                      storage_col, rng):
    """
    Duplica source_obj y lo coloca con la BASE pegada a la cara:
    - Centro X/Y = centro de la cara en espacio mundo.
    - Eje Z alineado con la normal de la cara.
    - El objeto se eleva por la mitad de su bounding box local en Z
      (eje local del objeto = normal de la cara) para que su base
      quede exactamente sobre la superficie, sin hundirse.
    """
    from mathutils import Vector

    new_obj      = source_obj.copy()
    new_obj.data = source_obj.data.copy()

    # ── Orientación: Z local = normal de la cara ──
    up    = normal_w
    x_ref = Vector((1, 0, 0))
    if abs(up.dot(x_ref)) > 0.99:
        x_ref = Vector((0, 1, 0))
    tx = (x_ref - x_ref.dot(up) * up).normalized()
    ty = up.cross(tx).normalized()
    tx = ty.cross(up).normalized()

    # ── Elevación: mínimo Z del bounding box local del fuente ──
    # El bounding box está en espacio local del objeto fuente.
    # Queremos que el punto más bajo (min Z local) quede en center_w.
    bbox_zs  = [v[2] for v in source_obj.bound_box]   # Z en espacio local
    z_min    = min(bbox_zs)   # puede ser negativo si el origen no está en la base
    z_offset = -z_min         # distancia a añadir para que min_Z quede en 0

    # Desplazar center_w a lo largo de la normal por z_offset
    final_pos = center_w + up * z_offset

    mat = Matrix((
        (tx.x, ty.x, up.x, final_pos.x),
        (tx.y, ty.y, up.y, final_pos.y),
        (tx.z, ty.z, up.z, final_pos.z),
        (0,    0,    0,    1           ),
    ))
    new_obj.matrix_world = mat

    # ── Etiquetas para limpiar después ──
    new_obj["mm_source_plane"] = context.active_object.name if context.active_object else ""
    new_obj["mm_level"]        = ""

    # ── Enlazar a la carpeta de almacenaje ──
    storage_col.objects.link(new_obj)
    for col in list(new_obj.users_collection):
        if col != storage_col:
            col.objects.unlink(new_obj)

    return new_obj


# ================================================================
#  CÁLCULO DE DISTANCIAS — BFS sobre la malla
# ================================================================

def build_face_adjacency(bm):
    """
    Devuelve un dict {face_index: [face_index, ...]} con las caras
    adyacentes (comparten una arista) para cada cara.
    """
    adj = {f.index: [] for f in bm.faces}
    for edge in bm.edges:
        linked = edge.link_faces
        if len(linked) == 2:
            a, b = linked[0].index, linked[1].index
            adj[a].append(b)
            adj[b].append(a)
    return adj


def bfs_distances(source_indices, adjacency):
    """
    BFS desde múltiples fuentes.
    Devuelve {face_index: distancia_minima_a_cualquier_fuente}.
    """
    dist  = {fi: 0 for fi in source_indices}
    queue = deque(source_indices)
    while queue:
        fi = queue.popleft()
        for nb in adjacency[fi]:
            if nb not in dist:
                dist[nb] = dist[fi] + 1
                queue.append(nb)
    return dist


# ================================================================
#  NÚCLEO — asignación de niveles y generación
# ================================================================

def assign_levels(plaza_indices, distances, seed):
    """
    Devuelve {face_index: nivel} para todas las caras del mapa.

    Reglas:
    - dist 0        → "plaza"
    - dist 1        → 1 cara por P será "iglesia" (aleatoria entre adyacentes),
                      el resto → "orden1"
    - dist 2        → "orden2"
    - dist 3        → "orden3"
    - dist >= 4     → "orden4"
    - caras sin dist (desconectadas) → "orden4"
    """
    rng        = random.Random(seed)
    assignment = {}

    # Construir mapa inverso: cara_de_dist1 → lista de plazas a dist 1
    # Para poder asignar exactamente 1 iglesia por plaza
    dist1_per_plaza = {}   # {plaza_index: [vecinos_de_dist_1]}

    # Necesitamos la adyacencia para saber los vecinos directos de cada P
    # Se pasa a través de distances: los vecinos de P con dist==1 son los candidatos
    # Reconstruimos: para cada P, sus vecinos directos con dist==1
    # (esto requiere la adyacencia; la pasamos como parámetro extra)
    # → ver llamador: assign_levels recibe adjacency también
    pass  # ver versión real abajo


def assign_levels_full(plaza_indices, distances, adjacency, seed, settings):
    """
    Asigna un nivel a cada cara según la distancia BFS desde las plazas.

    Los órdenes 1, 2 y 3 tienen un número de anillos configurable.
    Ejemplo con orden1.rings=1, orden2.rings=2, orden3.rings=1:
      dist 0        → plaza
      dist 1        → iglesia (1 por plaza) o orden1
      dist 2-3      → orden2   (2 anillos)
      dist 4        → orden3
      dist >= 5     → orden4

    Los rangos se calculan sumando los rings en orden:
      orden1: [1,            1 + rings1 - 1]
      orden2: [1 + rings1,   1 + rings1 + rings2 - 1]
      orden3: [1 + rings1 + rings2, ...]
    """
    rng        = random.Random(seed ^ 0xC1D7AD)
    assignment = {}

    # ── Calcular rangos de distancia para cada orden ──
    r1 = settings.orden1.rings
    r2 = settings.orden2.rings
    r3 = settings.orden3.rings

    # Inicio y fin (inclusive) de distancia para cada orden
    d1_start = 1;          d1_end = r1
    d2_start = d1_end + 1; d2_end = d1_end + r2
    d3_start = d2_end + 1; d3_end = d2_end + r3
    # orden4: cualquier dist > d3_end

    # ── Paso 1: plaza ──
    for fi in plaza_indices:
        assignment[fi] = "plaza"

    # ── Paso 2: iglesia — 1 vecino de dist==1 por cada plaza ──
    iglesia_set = set()
    for p_fi in plaza_indices:
        candidates = [
            nb for nb in adjacency[p_fi]
            if distances.get(nb) == 1 and nb not in iglesia_set
        ]
        if candidates:
            rng.shuffle(candidates)
            iglesia_set.add(candidates[0])
            assignment[candidates[0]] = "iglesia"

    # ── Paso 3: resto de caras por rango de distancia ──
    for fi, d in distances.items():
        if fi in assignment:
            continue
        if d1_start <= d <= d1_end:
            assignment[fi] = "orden1"
        elif d2_start <= d <= d2_end:
            assignment[fi] = "orden2"
        elif d3_start <= d <= d3_end:
            assignment[fi] = "orden3"
        else:
            assignment[fi] = "orden4"

    return assignment


# ================================================================
#  CARRETERAS — generación por aristas del grid
# ================================================================

# ================================================================
#  CARRETERAS — array de tiles por arista + tile de intersección
# ================================================================

def _link_to_storage(new_obj, storage):
    """Enlaza el objeto a storage y lo desvincula del resto."""
    storage.objects.link(new_obj)
    for col in list(new_obj.users_collection):
        if col != storage:
            col.objects.unlink(new_obj)


def _place_road_tile(source, mat_world, storage, plane_name):
    """Duplica source, aplica mat_world, etiqueta y enlaza a storage."""
    new_obj      = source.copy()
    new_obj.data = source.data.copy()
    new_obj.matrix_world = mat_world
    new_obj["mm_source_plane"] = plane_name
    new_obj["mm_level"]        = "road"
    _link_to_storage(new_obj, storage)
    return new_obj


def generate_roads(context, bm, obj, settings, storage, plane_name, rng):
    """
    Genera carreteras en dos pasadas:

    PASADA 1 — Segmentos (aristas interiores):
      Para cada arista interior con 2 caras adyacentes se decide si
      hay carretera (prob = road.density). Si la hay, se instancian
      copias del tile de segmento en array a lo largo de la arista,
      sin estirar el asset. La longitud del tile determina cuántas
      copias caben; si sobra espacio la última copia queda tal cual.

    PASADA 2 — Intersecciones (vértices):
      Un vértice recibe un tile de intersección si al menos UNA de
      sus aristas adyacentes generó segmento de carretera.
      La prob de aparición es independiente (intersection.density).
      El tile se orienta con el mismo sistema de ejes que los segmentos
      (Z = normal del plano) y se escala para cubrir el área del cruce,
      calculada como la media de las longitudes de sus aristas.
    """
    seg_col    = get_collection(settings.road.collection)
    seg_prob   = settings.road.density
    cross_col  = get_collection(settings.road_intersection.collection)
    cross_prob = settings.road_intersection.density

    mat_obj = obj.matrix_world
    total   = 0

    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    # Conjunto de aristas que SÍ generaron carretera (para la pasada 2)
    road_edges = set()

    # ── PASADA 1: segmentos ──
    if seg_col is not None and seg_prob > 0.0:

        # Calcular longitud del tile del asset (eje X local)
        seg_source_ref = [o for o in seg_col.objects if o.type == 'MESH']
        if seg_source_ref:
            ref_bbox = seg_source_ref[0].bound_box
            tile_len = max(v[0] for v in ref_bbox) - min(v[0] for v in ref_bbox)
            tile_len = max(tile_len, 1e-4)
        else:
            tile_len = 1.0

        for edge in bm.edges:
            linked = edge.link_faces
            if len(linked) < 2:          # arista de borde → sin carretera
                continue
            if rng.random() >= seg_prob:  # decisión probabilística
                continue

            source = random_obj_from_collection(seg_col, rng)
            if source is None:
                continue

            # Geometría de la arista en espacio mundo
            v0_w     = mat_obj @ edge.verts[0].co
            v1_w     = mat_obj @ edge.verts[1].co
            edge_vec = v1_w - v0_w
            edge_len = edge_vec.length
            if edge_len < 1e-6:
                continue
            edge_dir = edge_vec / edge_len

            # Normal media de las caras adyacentes → eje Z del tile
            normals = [(mat_obj.to_3x3() @ f.normal).normalized() for f in linked]
            up = sum(normals, normals[0].__class__()).normalized()

            # Base ortonormal: X = dirección arista, Y = up × X, Z = up
            ty = up.cross(edge_dir).normalized()
            tx = ty.cross(up).normalized()

            # Elevación: base del tile sobre la arista
            bbox_zs  = [v[2] for v in source.bound_box]
            z_offset = -min(bbox_zs)

            # Punto de inicio del array: extremo v0 + z_offset
            start = v0_w + up * z_offset

            # Número de copias que caben (sin fraccionar)
            n_tiles = max(1, int(edge_len / tile_len))

            for k in range(n_tiles):
                # Centrar cada tile en su posición a lo largo de la arista
                offset_along = (k + 0.5) * tile_len
                pos = start + tx * offset_along

                mat = Matrix((
                    (tx.x, ty.x, up.x, pos.x),
                    (tx.y, ty.y, up.y, pos.y),
                    (tx.z, ty.z, up.z, pos.z),
                    (0,    0,    0,    1     ),
                ))
                new_obj = _place_road_tile(source, mat, storage, plane_name)
                new_obj.name = f"MM_Seg_{new_obj.name}"
                total += 1

            road_edges.add(edge.index)

    # ── PASADA 2: intersecciones ──
    if cross_col is not None and cross_prob > 0.0 and road_edges:

        for vert in bm.verts:
            # ¿Alguna arista adyacente a este vértice generó carretera?
            # Incluye vértices de borde (conectados a 1-3 aristas)
            adjacent_road_edges = [e for e in vert.link_edges if e.index in road_edges]
            if not adjacent_road_edges:
                continue

            if rng.random() >= cross_prob:
                continue

            source = random_obj_from_collection(cross_col, rng)
            if source is None:
                continue

            # Posición: vértice en espacio mundo
            vert_w = mat_obj @ vert.co

            # Normal: media de las caras adyacentes al vértice
            linked_faces = vert.link_faces
            if not linked_faces:
                continue
            normals = [(mat_obj.to_3x3() @ f.normal).normalized() for f in linked_faces]
            up = sum(normals, normals[0].__class__()).normalized()

            # Orientación: usar la dirección de la primera arista de carretera
            # adyacente como referencia para el eje X
            ref_edge  = adjacent_road_edges[0]
            other_v   = ref_edge.other_vert(vert)
            ref_dir   = (mat_obj @ other_v.co - vert_w).normalized()
            ty = up.cross(ref_dir).normalized()
            tx = ty.cross(up).normalized()

            # Tamaño fijo del tile de intersección
            cross_size = 1.2

            # Escalar el tile de intersección al tamaño fijo
            bbox_xs  = [v[0] for v in source.bound_box]
            bbox_ys  = [v[1] for v in source.bound_box]
            bbox_zs  = [v[2] for v in source.bound_box]
            orig_w   = max(bbox_xs) - min(bbox_xs)
            orig_d   = max(bbox_ys) - min(bbox_ys)
            z_offset = -min(bbox_zs)
            sx = cross_size / orig_w if orig_w > 1e-6 else 1.0
            sy = cross_size / orig_d if orig_d > 1e-6 else 1.0

            pos = vert_w + up * z_offset

            mat = Matrix((
                (tx.x * sx, ty.x * sy, up.x, pos.x),
                (tx.y * sx, ty.y * sy, up.y, pos.y),
                (tx.z * sx, ty.z * sy, up.z, pos.z),
                (0,         0,         0,    1     ),
            ))

            new_obj = _place_road_tile(source, mat, storage, plane_name)
            new_obj.name = f"MM_Cruce_{new_obj.name}"
            total += 1

    return total


# ================================================================
#  PROPERTY GROUPS
# ================================================================

class MM_LevelSlot(PropertyGroup):
    """Un slot de colección para un nivel de ciudad."""
    collection: PointerProperty(
        name        = "Colección",
        type        = bpy.types.Collection,
        description = "Colección con los assets de este nivel",
    )
    rings: IntProperty(
        name        = "Anillos",
        description = "Número de anillos de distancia que ocupa este nivel "
                      "(solo para órdenes 1, 2 y 3; plaza, iglesia y orden4 son fijos)",
        default     = 1,
        min         = 1,
        max         = 20,
    )


class MM_RoadSlot(PropertyGroup):
    """Slot de segmento de carretera: tile que se repite en array."""
    collection: PointerProperty(
        name        = "Colección",
        type        = bpy.types.Collection,
        description = "Colección con los tiles de segmento recto de carretera",
    )
    density: FloatProperty(
        name        = "Probabilidad",
        description = "0 = nunca · 1 = siempre en todas las aristas",
        default     = 0.5, min=0.0, max=1.0, subtype='FACTOR',
    )


class MM_RoadIntersectionSlot(PropertyGroup):
    """Slot de tile de intersección/cruce de carretera."""
    collection: PointerProperty(
        name        = "Colección",
        type        = bpy.types.Collection,
        description = "Colección con los tiles de cruce/intersección",
    )
    density: FloatProperty(
        name        = "Probabilidad",
        description = "0 = nunca · 1 = en todos los vértices con carretera adyacente",
        default     = 0.8, min=0.0, max=1.0, subtype='FACTOR',
    )


class MM_SceneSettings(PropertyGroup):
    seed: IntProperty(
        name="Seed", description="Semilla aleatoria",
        default=0, min=0, max=999999,
    )
    storage: PointerProperty(
        name="Carpeta de almacenaje",
        type=bpy.types.Collection,
        description="Colección donde se guardarán los edificios generados",
    )
    plaza             : PointerProperty(type=MM_LevelSlot)
    iglesia           : PointerProperty(type=MM_LevelSlot)
    orden1            : PointerProperty(type=MM_LevelSlot)
    orden2            : PointerProperty(type=MM_LevelSlot)
    orden3            : PointerProperty(type=MM_LevelSlot)
    orden4            : PointerProperty(type=MM_LevelSlot)
    road              : PointerProperty(type=MM_RoadSlot)
    road_intersection : PointerProperty(type=MM_RoadIntersectionSlot)

    # ── Estado de secciones colapsables ──
    open_config  : BoolProperty(name="Configuración",  default=True)
    open_niveles : BoolProperty(name="Assets por nivel", default=True)
    open_carret  : BoolProperty(name="Carreteras",      default=False)


# ================================================================
#  OPERADOR — Generar ciudad
# ================================================================

class MM_OT_generate(bpy.types.Operator):
    """Genera la ciudad sobre el plano activo a partir de las caras seleccionadas como plaza."""
    bl_idname  = "makimalum.generate"
    bl_label   = "Generar ciudad"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (obj is not None
                and obj.type == 'MESH'
                and obj.mode in {'EDIT', 'OBJECT'})

    def execute(self, context):
        settings = context.scene.mm_settings
        obj      = context.active_object
        rng      = random.Random(settings.seed)

        # ── Colección de almacenaje ──
        storage = get_collection(settings.storage)
        if storage is None:
            self.report({'ERROR'}, "Selecciona una Carpeta de Almacenaje antes de generar")
            return {'CANCELLED'}

        # ── Leer caras seleccionadas como plazas ──
        was_edit = (obj.mode == 'EDIT')
        if was_edit:
            bpy.ops.object.mode_set(mode='OBJECT')

        mesh = obj.data
        bm   = bmesh.new()
        bm.from_mesh(mesh)
        bm.faces.ensure_lookup_table()

        plaza_indices = [f.index for f in bm.faces if f.select]

        if not plaza_indices:
            bm.free()
            if was_edit:
                bpy.ops.object.mode_set(mode='EDIT')
            self.report({'WARNING'}, "Selecciona al menos una cara como Plaza Central")
            return {'CANCELLED'}

        # ── BFS desde las plazas ──
        adjacency = build_face_adjacency(bm)
        distances  = bfs_distances(plaza_indices, adjacency)

        # ── Asignar niveles ──
        assignment = assign_levels_full(
            plaza_indices, distances, adjacency, settings.seed, settings
        )
        # Caras no alcanzadas por BFS (malla no totalmente conectada) → orden4
        for face in bm.faces:
            if face.index not in assignment:
                assignment[face.index] = "orden4"

        # ── Eliminar objetos generados anteriormente por este plano ──
        plane_name = obj.name
        old_objs = [
            o for o in bpy.data.objects
            if o.get("mm_source_plane") == plane_name
        ]
        for o in old_objs:
            bpy.data.objects.remove(o, do_unlink=True)

        # ── Generar props por nivel (orden de prioridad) ──
        total = 0
        mat_obj = obj.matrix_world

        for level in LEVELS:
            slot_data = getattr(settings, level)
            col       = get_collection(slot_data.collection)
            if col is None:
                continue

            for fi, lvl in assignment.items():
                if lvl != level:
                    continue

                face     = bm.faces[fi]
                source   = random_obj_from_collection(col, rng)
                if source is None:
                    continue

                center_w = face_center_world(face, mat_obj)
                normal_w = face_normal_world(face, mat_obj)

                new_obj = place_obj_on_face(
                    context, source, center_w, normal_w, storage, rng
                )
                new_obj["mm_source_plane"] = plane_name
                new_obj["mm_level"]        = level
                new_obj.name = f"MM_{LEVEL_LABELS.get(level,'obj')}_{new_obj.name}"
                total += 1

        # ── Generar carreteras por aristas ──
        road_count = generate_roads(
            context, bm, obj, settings, storage, plane_name, rng
        )
        total += road_count

        bm.free()

        if was_edit:
            bpy.ops.object.mode_set(mode='EDIT')

        self.report({'INFO'}, f"MakiMalum: {total} objeto(s) generado(s)")
        return {'FINISHED'}


# ================================================================
#  OPERADOR — Limpiar ciudad generada
# ================================================================

class MM_OT_clear(bpy.types.Operator):
    """Elimina todos los edificios generados por MakiMalum del objeto activo."""
    bl_idname  = "makimalum.clear"
    bl_label   = "Limpiar ciudad"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH'

    def execute(self, context):
        plane_name = context.active_object.name
        old_objs   = [
            o for o in bpy.data.objects
            if o.get("mm_source_plane") == plane_name
        ]
        count = len(old_objs)
        for o in old_objs:
            bpy.data.objects.remove(o, do_unlink=True)
        self.report({'INFO'}, f"MakiMalum: {count} objeto(s) eliminado(s)")
        return {'FINISHED'}


# ================================================================
#  PANEL PRINCIPAL
# ================================================================

def mm_section_header(layout, settings, open_attr, label, icon='DOT'):
    """Cabecera colapsable para el panel de MakiMalum."""
    row = layout.row()
    row.prop(settings, open_attr,
             icon='TRIA_DOWN' if getattr(settings, open_attr) else 'TRIA_RIGHT',
             icon_only=True, emboss=False)
    row.label(text=label, icon=icon)
    return getattr(settings, open_attr)


class MM_PT_main(bpy.types.Panel):
    bl_label       = "MakiMalum"
    bl_idname      = "MM_PT_main"
    bl_space_type  = "VIEW_3D"
    bl_region_type = "UI"
    bl_category    = "MakiMalum"

    def draw(self, context):
        layout   = self.layout
        settings = context.scene.mm_settings
        obj      = context.active_object

        if obj is None or obj.type != 'MESH':
            layout.label(text="Selecciona un plano Mesh", icon='ERROR')
            return

        # ── Configuración ──
        box = layout.box()
        if mm_section_header(box, settings, "open_config", "Configuración", 'PREFERENCES'):
            box.prop(settings, "seed",    text="Semilla")
            box.prop(settings, "storage", text="Carpeta de almacenaje")

        layout.separator(factor=0.5)

        # ── Assets por nivel ──
        box = layout.box()
        if mm_section_header(box, settings, "open_niveles", "Assets por nivel", 'COMMUNITY'):
            for level in LEVELS:
                slot   = getattr(settings, level)
                col_ui = box.column(align=False)
                row    = col_ui.row(align=True)
                row.label(text=LEVEL_LABELS[level], icon=LEVEL_ICONS[level])
                row.prop(slot, "collection", text="")
                if level in ("orden1", "orden2", "orden3"):
                    sub = col_ui.row(align=True)
                    sub.label(text="")
                    sub.prop(slot, "rings", text="Anillos")

        layout.separator(factor=0.5)

        # ── Carreteras ──
        box = layout.box()
        if mm_section_header(box, settings, "open_carret", "Carreteras", 'DRIVER_DISTANCE'):
            sub = box.column(align=True)
            sub.label(text="Segmento (tile recto):", icon='CURVE_PATH')
            sub.prop(settings.road, "collection", text="Colección")
            sub.prop(settings.road, "density",    text="Probabilidad", slider=True)
            box.separator()
            sub2 = box.column(align=True)
            sub2.label(text="Intersección (tile de cruce):", icon='OUTLINER_OB_LATTICE')
            sub2.prop(settings.road_intersection, "collection", text="Colección")
            sub2.prop(settings.road_intersection, "density",    text="Probabilidad", slider=True)

        layout.separator(factor=0.5)

        in_edit = (obj.mode == 'EDIT')
        if in_edit:
            layout.label(text="Selecciona las caras plaza y genera:", icon='INFO')
        else:
            layout.label(text="Entra en EditMode para seleccionar caras plaza", icon='INFO')

        col = layout.column(align=True)
        col.scale_y = 1.5
        col.operator("makimalum.generate", text="Generar ciudad", icon='WORLD')
        col.operator("makimalum.clear",    text="Limpiar ciudad", icon='TRASH')


# ================================================================
#  REGISTRO
# ================================================================

classes = [
    MM_LevelSlot,
    MM_RoadSlot,
    MM_RoadIntersectionSlot,
    MM_SceneSettings,
    MM_OT_generate,
    MM_OT_clear,
    MM_PT_main,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mm_settings = PointerProperty(type=MM_SceneSettings)


def unregister():
    del bpy.types.Scene.mm_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
