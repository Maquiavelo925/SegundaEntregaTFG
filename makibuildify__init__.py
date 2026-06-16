# ================================================================
#  MakiBuildify — Generador procedural de fachadas de edificio
#  Autor   : Makiavelo925, Álex Cuevas Langa
#  Versión : 1.4.0  |  Blender 4.2+
#
#  v1.4 — Generación desde el perímetro hacia afuera:
#  1. Extraer el contorno exterior del plano en orden (Shoelace
#     garantiza orientación antihoraria para que las normales
#     apunten siempre hacia afuera).
#  2. Por cada vértice del contorno: colocar esquina centrada.
#     El tipo (ext/int) se determina por el signo del ángulo girado
#     en ese vértice (cross product de los dos segmentos adyacentes).
#  3. Por cada segmento entre dos vértices: rellenar con paredes.
#     n = floor(espacio_libre / 3m), ancho_ajustado = espacio / n.
#  4. Todo apunta hacia afuera usando la normal del segmento
#     (perpendicular al segmento, en el lado exterior del polígono).
# ================================================================

bl_info = {
    "name":        "MakiBuildify",
    "author":      "Makiavelo925, Álex Cuevas Langa",
    "version":     (1, 6, 0),
    "blender":     (4, 2, 0),
    "location":    "View3D › N-Panel › MakiBuildify",
    "description": "Genera fachadas de edificio procedurales sobre un plano de planta",
    "category":    "Add Mesh",
}

import bpy
import bmesh
import random
import math
from mathutils import Vector, Matrix
from bpy.props import (
    IntProperty, FloatProperty,
    BoolProperty, PointerProperty,
    StringProperty,
)
from bpy.types import PropertyGroup


# ================================================================
#  CONSTANTES
# ================================================================

MODULE_WIDTH_REF  = 3.0
MODULE_HEIGHT_REF = 3.0
PLANE_ORIG_Z_KEY  = "mbf_orig_z"


# ================================================================
#  UTILIDADES — colecciones
# ================================================================

def collection_items(self, context):
    """Mantenido solo para compatibilidad con storage si fuera necesario."""
    items = [("NONE", "— Ninguna —", "")]
    for col in bpy.data.collections:
        items.append((col.name, col.name, ""))
    return items

def get_col(col_or_name):
    """
    Acepta tanto una Collection de Blender (PointerProperty)
    como un string con el nombre (compatibilidad).
    """
    if col_or_name is None:
        return None
    if isinstance(col_or_name, str):
        if col_or_name == "NONE" or col_or_name == "":
            return None
        return bpy.data.collections.get(col_or_name)
    # Es directamente un objeto Collection
    return col_or_name

def rand_obj(col, rng):
    if col is None:
        return None
    objs = [o for o in col.objects if o.type == 'MESH']
    return rng.choice(objs) if objs else None


# ================================================================
#  UTILIDADES — geometría de módulos
# ================================================================

def bbox_dims(obj):
    b = obj.bound_box
    xs=[v[0] for v in b]; ys=[v[1] for v in b]; zs=[v[2] for v in b]
    return max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)

def z_base_offset(obj):
    """Desplazamiento Z para que la BASE del objeto quede en z=0 local."""
    return -min(v[2] for v in obj.bound_box)

def z_center_offset(obj):
    """Desplazamiento Z para que el CENTRO del objeto quede en z=0 local."""
    zs = [v[2] for v in obj.bound_box]
    return -(min(zs) + max(zs)) * 0.5

def place_module(source, mat, storage, plane_name, label):
    new = source.copy()
    new.data = source.data.copy()
    new.matrix_world = mat
    new["mbf_plane"] = plane_name
    new["mbf_label"] = label
    new.name = f"MBF_{label}_{new.name}"
    storage.objects.link(new)
    for c in list(new.users_collection):
        if c != storage:
            c.objects.unlink(new)
    return new

def mod_matrix(pos, tx, ty, tz, sx=1.0, sy=1.0, sz=1.0):
    return Matrix((
        (tx.x*sx, ty.x*sy, tz.x*sz, pos.x),
        (tx.y*sx, ty.y*sy, tz.y*sz, pos.y),
        (tx.z*sx, ty.z*sy, tz.z*sz, pos.z),
        (0,       0,       0,       1     ),
    ))


# ================================================================
#  EXTRACCIÓN DEL CONTORNO EXTERIOR ORDENADO
# ================================================================

def extract_perimeter(bm, obj_matrix):
    """
    Devuelve la lista de posiciones mundo del contorno exterior del plano
    en orden antihorario (visto desde arriba), calculada recorriendo las
    aristas de borde en secuencia.

    Retorna: lista de Vector (posiciones mundo), lista de índices de vértice.
    El último punto NO se repite (el contorno es implícitamente cerrado).
    """
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    border_edges = [e for e in bm.edges if len(e.link_faces) == 1]
    if not border_edges:
        return [], []

    # Mapa vértice → aristas de borde adyacentes
    v2e = {}
    for e in border_edges:
        for v in e.verts:
            v2e.setdefault(v.index, []).append(e)

    # Recorrer el contorno ordenadamente
    start_edge = border_edges[0]
    cur_vert   = start_edge.verts[0]
    cur_edge   = start_edge
    ordered_vi = [cur_vert.index]
    visited_e  = {cur_edge.index}

    for _ in range(len(border_edges)):
        next_e = None
        for e in v2e.get(cur_vert.index, []):
            if e.index not in visited_e:
                next_e = e
                break
        if next_e is None:
            break
        cur_vert = next_e.other_vert(cur_vert)
        cur_edge = next_e
        visited_e.add(next_e.index)
        if cur_vert.index == ordered_vi[0]:
            break   # contorno cerrado
        ordered_vi.append(cur_vert.index)

    if len(ordered_vi) < 3:
        return [], []

    pts_w = [obj_matrix @ bm.verts[vi].co for vi in ordered_vi]

    # Asegurar orientación antihoraria (Shoelace):
    # área con signo positivo → antihorario en XY con Z hacia arriba
    area2 = 0.0
    n = len(pts_w)
    for i in range(n):
        j = (i + 1) % n
        area2 += pts_w[i].x * pts_w[j].y - pts_w[j].x * pts_w[i].y
    if area2 < 0:
        pts_w.reverse()
        ordered_vi.reverse()

    return pts_w, ordered_vi


# ================================================================
#  CLASIFICACIÓN DE VÉRTICE (convex / concave)
# ================================================================

def corner_type_at(pts_w, i):
    """
    Determina si el vértice i del contorno (antihorario) es
    convexo (exterior del edificio) o cóncavo (interior/recodo).

    Con orientación antihoraria:
      cross_z > 0 → giro a la izquierda → vértice convexo (ext)
      cross_z < 0 → giro a la derecha  → vértice cóncavo  (int)

    También devuelve (tx, ty) para orientar el asset de esquina:
      ty = bisectriz de las normales de los dos segmentos adyacentes
           (apunta hacia afuera del edificio)
      tx = tz × ty
    """
    n   = len(pts_w)
    tz  = Vector((0, 0, 1))
    prv = pts_w[(i - 1) % n]
    cur = pts_w[i]
    nxt = pts_w[(i + 1) % n]

    # Vectores de los dos segmentos adyacentes al vértice
    d0 = (cur - prv).normalized()   # llega al vértice
    d1 = (nxt - cur).normalized()   # sale del vértice

    # Normales de cada segmento (perpendicular, hacia la izquierda = exterior
    # en contorno antihorario)
    n0 = Vector((-d0.y, d0.x, 0)).normalized()
    n1 = Vector((-d1.y, d1.x, 0)).normalized()

    # Clasificación por cross product en Z de d0 y d1
    cross_z = d0.x * d1.y - d0.y * d1.x
    ctype   = 'ext' if cross_z > 0 else 'int'

    # Orientación: bisectriz de las dos normales (apunta hacia afuera)
    bis = (n0 + n1)
    if bis.length < 1e-6:
        bis = n0.copy()
    bis_n = bis.normalized()

    tx = tz.cross(bis_n).normalized()
    ty = tx.cross(tz).normalized()   # re-ortogonalizar
    return ctype, tx, ty


def segment_normal(p0, p1):
    """
    Normal del segmento p0→p1 apuntando hacia el exterior del polígono
    (a la izquierda del segmento en contorno antihorario).
    """
    d = (p1 - p0)
    d.z = 0
    if d.length < 1e-6:
        return Vector((0, 1, 0))
    d = d.normalized()
    return Vector((-d.y, d.x, 0)).normalized()


# ================================================================
#  RELLENO DE PAREDES ENTRE ESQUINAS
# ================================================================

def walls_fill(seg_len, esq_half_w, ref=MODULE_WIDTH_REF):
    """
    Calcula cuántas paredes caben en el espacio libre entre dos esquinas
    y el ancho ajustado de cada una.

    espacio_libre = seg_len - 2 * esq_half_w
    n = max(1, floor(espacio_libre / ref))
    ancho = espacio_libre / n

    Si el espacio libre es ≤ 0 no se generan paredes (solo esquinas).
    """
    free = seg_len - 2.0 * esq_half_w
    if free <= 1e-3:
        return 0, 0.0
    n = max(1, int(free / ref))
    return n, free / n


# ================================================================
#  DECORACIONES — helper excluyente
# ================================================================

def apply_deco_group(grp, col_center, tx, ty, tz,
                     storage, plane_name, rng, label):
    col = get_col(grp.collection)
    if col is None or rng.random() >= grp.density:
        return 0
    src = rand_obj(col, rng)
    if src is None:
        return 0
    jx  = rng.uniform(grp.jitter_x_min, grp.jitter_x_max)
    jz  = rng.uniform(grp.jitter_z_min, grp.jitter_z_max)
    # Usar z_center_offset para que el CENTRO del objeto quede en col_center,
    # no la base. Con jitter Z = 0 la decoración queda exactamente centrada
    # en la altura del módulo de piso.
    zo  = z_center_offset(src)
    pos = col_center + tx * jx + tz * (jz + zo)
    mat = mod_matrix(pos, tx, ty, tz)
    place_module(src, mat, storage, plane_name, label)
    return 1


# ================================================================
#  GENERACIÓN DESDE EL PERÍMETRO
# ================================================================

def generate_from_perimeter(context, bm, obj, settings, storage, plane_name, rng):
    """
    Recorre el contorno exterior del plano segmento a segmento.
    Por cada vértice: coloca la columna de esquina (base+pisos+cornisa).
    Por cada segmento: rellena con columnas de pared entre las esquinas.
    """
    obj_matrix = obj.matrix_world
    pts_w, vis = extract_perimeter(bm, obj_matrix)
    if not pts_w:
        return 0

    n_pts = len(pts_w)
    tz    = Vector((0, 0, 1))
    fh    = MODULE_HEIGHT_REF
    n_fl  = settings.n_floors
    total = 0

    # Colecciones
    base_col    = get_col(settings.base.collection)
    piso_col    = get_col(settings.piso.collection)
    cornisa_col = get_col(settings.cornisa.collection)
    pilar_col   = get_col(settings.pilar.collection)
    pilar_dens  = settings.pilar.density

    cols_esq = {
        'ext': (get_col(settings.esq_ext_base.collection),
                get_col(settings.esq_ext_piso.collection),
                get_col(settings.esq_ext_cornisa.collection)),
        'int': (get_col(settings.esq_int_base.collection),
                get_col(settings.esq_int_piso.collection),
                get_col(settings.esq_int_cornisa.collection)),
    }

    # Calcular ancho de mitad de esquina (del bounding box del asset)
    # para descontar el espacio que ocupa en cada segmento.
    # Usamos el asset de esquina exterior piso como referencia (el más común).
    esq_ref_col = cols_esq['ext'][1] or cols_esq['int'][1]
    esq_half_w  = 0.0
    if esq_ref_col:
        esq_ref_src = rand_obj(esq_ref_col, random.Random(0))
        if esq_ref_src:
            bw, _, _ = bbox_dims(esq_ref_src)
            esq_half_w = bw * 0.5

    # ── RECORRER VÉRTICES Y SEGMENTOS ──
    for i in range(n_pts):
        cur = pts_w[i]
        nxt = pts_w[(i + 1) % n_pts]

        # ── 1. ESQUINA en el vértice i ──
        ctype, tx_e, ty_e = corner_type_at(pts_w, i)
        esq_cols = cols_esq[ctype]

        for fi in range(n_fl + 2):
            if fi == 0:
                col = esq_cols[0]; lbl = f"esq_{ctype}_base"
            elif fi == n_fl + 1:
                col = esq_cols[2]; lbl = f"esq_{ctype}_cornisa"
            else:
                col = esq_cols[1]; lbl = f"esq_{ctype}_piso"

            src = rand_obj(col, rng)
            if src is None:
                continue
            zo  = z_base_offset(src)
            pos = cur + tz * (fi * fh + zo)
            mat = mod_matrix(pos, tx_e, ty_e, tz)
            place_module(src, mat, storage, plane_name, lbl)
            total += 1

        # ── 2. PAREDES en el segmento i → i+1 ──
        seg_vec = nxt - cur
        seg_len = seg_vec.length
        if seg_len < 1e-6:
            continue

        tx_w = seg_vec.normalized()   # a lo largo del segmento
        tx_w = (tx_w - tx_w.dot(tz) * tz).normalized()
        ty_w = segment_normal(cur, nxt)   # hacia fuera

        n_walls, wall_w = walls_fill(seg_len, esq_half_w)
        if n_walls == 0:
            continue

        # Origen de las paredes: después de la mitad de esquina
        wall_origin = cur + tx_w * esq_half_w

        for wi in range(n_walls):
            col_center_along = (wi + 0.5) * wall_w
            col_base_w = wall_origin + tx_w * col_center_along

            # ¿Esta columna individual tiene pilar?
            # La decisión se toma por columna (no por segmento) para
            # evitar rachas donde un segmento entero recibe pilares.
            has_pilar  = (pilar_col is not None
                          and pilar_dens > 0
                          and rng.random() < pilar_dens)
            pilar_src  = rand_obj(pilar_col, rng) if has_pilar else None

            for fi in range(n_fl + 2):
                if fi == 0:
                    lbl = "base";    col = base_col
                elif fi == n_fl + 1:
                    lbl = "cornisa"; col = cornisa_col
                else:
                    lbl = "piso";    col = piso_col

                src = rand_obj(col, rng)
                if src is None:
                    continue
                bw, _, _ = bbox_dims(src)
                sx  = wall_w / bw if bw > 1e-6 else 1.0
                zo  = z_base_offset(src)
                pos = col_base_w + tz * (fi * fh + zo)
                mat = mod_matrix(pos, tx_w, ty_w, tz, sx=sx, sy=1.0, sz=1.0)
                place_module(src, mat, storage, plane_name, lbl)
                total += 1

                # ── Decoraciones (solo si no hay pilar) ──
                if has_pilar:
                    continue

                mod_center = col_base_w + tz * (fi * fh + fh * 0.5)
                if fi == 0:
                    deco_groups = [settings.deco_base_a,
                                   settings.deco_base_b,
                                   settings.deco_base_c]
                    dlabel = "deco_base"
                elif fi == n_fl + 1:
                    deco_groups = [settings.deco_cornisa_a,
                                   settings.deco_cornisa_b,
                                   settings.deco_cornisa_c]
                    dlabel = "deco_cornisa"
                else:
                    deco_groups = [settings.deco_piso_a,
                                   settings.deco_piso_b,
                                   settings.deco_piso_c]
                    dlabel = "deco_piso"

                shuffled = list(deco_groups)
                rng.shuffle(shuffled)
                for grp in shuffled:
                    placed = apply_deco_group(
                        grp, mod_center, tx_w, ty_w, tz,
                        storage, plane_name, rng, dlabel
                    )
                    total += placed
                    if placed:
                        break

            # ── Pilares: array Z por columna de pared ──
            if pilar_src is not None:
                zo_p = z_base_offset(pilar_src)
                for fi in range(n_fl + 1):
                    pos = col_base_w + tz * (fi * fh + zo_p)
                    mat = mod_matrix(pos, tx_w, ty_w, tz)
                    place_module(pilar_src, mat, storage, plane_name, "pilar")
                    total += 1

    return total


# ================================================================
#  PROPERTY GROUPS
# ================================================================

class MBF_SlotProps(PropertyGroup):
    # PointerProperty a Collection en lugar de EnumProperty:
    # guarda la referencia real al objeto colección, no su índice.
    # Así no se desajusta cuando el usuario crea nuevas colecciones.
    collection: PointerProperty(
        name="Colección",
        type=bpy.types.Collection,
        description="Colección con los módulos de este slot",
    )

class MBF_PilarProps(PropertyGroup):
    collection: PointerProperty(
        name="Colección",
        type=bpy.types.Collection,
        description="Colección con los módulos de pilar",
    )
    density: FloatProperty(
        name="Densidad", default=0.3, min=0.0, max=1.0, subtype='FACTOR',
        description="Probabilidad de pilar por columna de pared",
    )

class MBF_DecoGroupProps(PropertyGroup):
    collection: PointerProperty(
        name="Colección",
        type=bpy.types.Collection,
    )
    density: FloatProperty(
        name="Densidad", default=0.3, min=0.0, max=1.0, subtype='FACTOR',
    )
    jitter_x_max: FloatProperty(name="Desp. X máx", default=0.5, min=-10.0, max=10.0, unit='LENGTH',
        description="Desplazamiento aleatorio máximo en el eje X de la fachada (permite negativos)")
    jitter_x_min: FloatProperty(name="Desp. X mín", default=-0.5, min=-10.0, max=10.0, unit='LENGTH',
        description="Desplazamiento aleatorio mínimo en el eje X de la fachada (permite negativos)")
    jitter_z_max: FloatProperty(name="Desp. Z máx", default=0.3, min=-10.0, max=10.0, unit='LENGTH',
        description="Desplazamiento aleatorio máximo en altura (permite negativos)")
    jitter_z_min: FloatProperty(name="Desp. Z mín", default=-0.3, min=-10.0, max=10.0, unit='LENGTH',
        description="Desplazamiento aleatorio mínimo en altura (permite negativos)")

class MBF_SceneSettings(PropertyGroup):
    seed: IntProperty(name="Semilla", default=0, min=0, max=999999)
    n_floors: IntProperty(
        name="Pisos", default=3, min=1, max=30,
        description="Número de pisos entre la base y la cornisa",
    )
    storage: PointerProperty(
        name="Carpeta de almacenaje",
        type=bpy.types.Collection,
        description="Colección donde se guardan los módulos generados",
    )

    base    : PointerProperty(type=MBF_SlotProps)
    piso    : PointerProperty(type=MBF_SlotProps)
    cornisa : PointerProperty(type=MBF_SlotProps)

    esq_ext_base    : PointerProperty(type=MBF_SlotProps)
    esq_ext_piso    : PointerProperty(type=MBF_SlotProps)
    esq_ext_cornisa : PointerProperty(type=MBF_SlotProps)
    esq_int_base    : PointerProperty(type=MBF_SlotProps)
    esq_int_piso    : PointerProperty(type=MBF_SlotProps)
    esq_int_cornisa : PointerProperty(type=MBF_SlotProps)

    pilar : PointerProperty(type=MBF_PilarProps)

    deco_base_a    : PointerProperty(type=MBF_DecoGroupProps)
    deco_base_b    : PointerProperty(type=MBF_DecoGroupProps)
    deco_base_c    : PointerProperty(type=MBF_DecoGroupProps)
    deco_piso_a    : PointerProperty(type=MBF_DecoGroupProps)
    deco_piso_b    : PointerProperty(type=MBF_DecoGroupProps)
    deco_piso_c    : PointerProperty(type=MBF_DecoGroupProps)
    deco_cornisa_a : PointerProperty(type=MBF_DecoGroupProps)
    deco_cornisa_b : PointerProperty(type=MBF_DecoGroupProps)
    deco_cornisa_c : PointerProperty(type=MBF_DecoGroupProps)

    open_config  : BoolProperty(name="Configuración",   default=True)
    open_fachada : BoolProperty(name="Fachada",          default=True)
    open_esq_ext : BoolProperty(name="Esq. Exteriores",  default=False)
    open_esq_int : BoolProperty(name="Esq. Interiores",  default=False)
    open_pilar   : BoolProperty(name="Pilares",          default=False)
    open_db      : BoolProperty(name="Deco Base",        default=False)
    open_dp      : BoolProperty(name="Deco Pisos",       default=False)
    open_dc      : BoolProperty(name="Deco Cornisa",     default=False)
    open_corrector : BoolProperty(name="Corrector de ángulos", default=False)

    # ── Parámetros del corrector de ángulos ──
    ac_target_angle : FloatProperty(
        name        = "Ángulo objetivo",
        description = "Ángulo interno al que se intentará llevar cada vértice",
        default     = 90.0, min=1.0, max=179.0, subtype='ANGLE',
        unit        = 'ROTATION',
    )
    ac_tolerance    : FloatProperty(
        name        = "Tolerancia",
        description = "Margen aceptable alrededor del ángulo objetivo (±)",
        default     = 5.0,  min=0.1, max=45.0, subtype='ANGLE',
        unit        = 'ROTATION',
    )
    ac_max_iters    : IntProperty(
        name        = "Iteraciones máx.",
        description = "Número máximo de pasadas de corrección",
        default     = 20, min=1, max=100,
    )


# ================================================================
#  HELPERS UI
# ================================================================

def section_header(layout, settings, open_attr, label, icon='DOT'):
    row = layout.row()
    row.prop(settings, open_attr,
             icon='TRIA_DOWN' if getattr(settings, open_attr) else 'TRIA_RIGHT',
             icon_only=True, emboss=False)
    row.label(text=label, icon=icon)
    return getattr(settings, open_attr)

def draw_deco_group(layout, grp, label):
    box = layout.box()
    box.label(text=label, icon='FUND')
    box.prop(grp, "collection", text="Colección")
    box.prop(grp, "density",    text="Densidad", slider=True)
    col = box.column(align=True)
    col.label(text="Desplazamiento eje X (fachada):")
    row = col.row(align=True)
    row.prop(grp, "jitter_x_min", text="Mín")
    row.prop(grp, "jitter_x_max", text="Máx")
    col.label(text="Desplazamiento eje Z (altura):")
    row = col.row(align=True)
    row.prop(grp, "jitter_z_min", text="Mín")
    row.prop(grp, "jitter_z_max", text="Máx")

def draw_corner_section(layout, settings, open_attr, label,
                        slot_base, slot_piso, slot_cornisa):
    box = layout.box()
    if section_header(box, settings, open_attr, label, icon='SNAP_VERTEX'):
        inner = box.column(align=True)
        for slot, lbl in [(slot_base,"Base"),(slot_piso,"Piso"),(slot_cornisa,"Cornisa")]:
            row = inner.row(align=True)
            row.label(text=lbl)
            row.prop(slot, "collection", text="")


# ================================================================
#  OPERADOR — Generar edificio
# ================================================================

class MBF_OT_generate(bpy.types.Operator):
    """Genera el edificio alrededor del perímetro del plano activo."""
    bl_idname  = "makibuildify.generate"
    bl_label   = "Generar edificio"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH'

    def execute(self, context):
        settings   = context.scene.mbf_settings
        obj        = context.active_object
        plane_name = obj.name
        rng        = random.Random(settings.seed)

        storage = get_col(settings.storage)
        if storage is None:
            self.report({'ERROR'}, "Selecciona una Carpeta de Almacenaje")
            return {'CANCELLED'}

        was_edit = (obj.mode == 'EDIT')
        if was_edit:
            bpy.ops.object.mode_set(mode='OBJECT')

        if PLANE_ORIG_Z_KEY not in obj:
            obj[PLANE_ORIG_Z_KEY] = obj.location.z

        old = [o for o in bpy.data.objects if o.get("mbf_plane") == plane_name]
        for o in old:
            bpy.data.objects.remove(o, do_unlink=True)

        bm = bmesh.new()
        bm.from_mesh(obj.data)

        total = generate_from_perimeter(
            context, bm, obj, settings, storage, plane_name, rng
        )
        bm.free()

        if total == 0:
            self.report({'WARNING'}, "No se encontró perímetro válido en el objeto")
            if was_edit:
                bpy.ops.object.mode_set(mode='EDIT')
            return {'CANCELLED'}

        roof_z = obj[PLANE_ORIG_Z_KEY] + (settings.n_floors + 1) * MODULE_HEIGHT_REF
        obj.location.z = roof_z

        if was_edit:
            bpy.ops.object.mode_set(mode='EDIT')

        self.report({'INFO'}, f"MakiBuildify: {total} módulo(s) generado(s)")
        return {'FINISHED'}


# ================================================================
#  OPERADOR — Limpiar edificio
# ================================================================

class MBF_OT_clear(bpy.types.Operator):
    """Elimina los módulos y restaura el plano a su posición original."""
    bl_idname  = "makibuildify.clear"
    bl_label   = "Limpiar edificio"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH'

    def execute(self, context):
        obj  = context.active_object
        name = obj.name
        old  = [o for o in bpy.data.objects if o.get("mbf_plane") == name]
        for o in old:
            bpy.data.objects.remove(o, do_unlink=True)
        if PLANE_ORIG_Z_KEY in obj:
            obj.location.z = obj[PLANE_ORIG_Z_KEY]
            del obj[PLANE_ORIG_Z_KEY]
        self.report({'INFO'}, f"MakiBuildify: {len(old)} módulo(s) eliminado(s)")
        return {'FINISHED'}


# ================================================================
#  CORRECTOR DE ÁNGULOS — integrado desde Buildify Angle Corrector v2
# ================================================================

def _internal_angle_at_vertex(v, face):
    """Ángulo interno en grados del vértice v dentro de face."""
    verts = list(face.verts)
    n     = len(verts)
    for i, vert in enumerate(verts):
        if vert != v:
            continue
        prev_v = verts[(i - 1) % n]
        next_v = verts[(i + 1) % n]
        vec_a  = (prev_v.co - v.co).normalized()
        vec_b  = (next_v.co - v.co).normalized()
        dot    = max(-1.0, min(1.0, vec_a.dot(vec_b)))
        return math.degrees(math.acos(dot))
    return None


def _corrected_position(v, face, desired_deg):
    """
    Calcula la nueva posición de v para que su ángulo interno sea
    desired_deg, manteniendo fijos los vecinos. Usa búsqueda binaria
    a lo largo de la bisectriz del ángulo actual.
    """
    verts = list(face.verts)
    n     = len(verts)
    for i, vert in enumerate(verts):
        if vert != v:
            continue
        prev_v = verts[(i - 1) % n]
        next_v = verts[(i + 1) % n]
        p = prev_v.co.copy()
        q = next_v.co.copy()
        c = v.co.copy()

        dir_a    = (p - c).normalized()
        dir_b    = (q - c).normalized()
        bisector = (dir_a + dir_b)
        if bisector.length < 1e-6:
            return None
        bisector = bisector.normalized()

        def angle_at(d):
            nc = c + bisector * d
            da = p - nc; db = q - nc
            if da.length < 1e-8 or db.length < 1e-8:
                return 0.0
            dot = max(-1.0, min(1.0, da.normalized().dot(db.normalized())))
            return math.degrees(math.acos(dot))

        max_dist      = ((p - c).length + (q - c).length) * 0.5
        current_angle = angle_at(0.0)
        eps           = max_dist * 0.001
        going_up      = angle_at(eps) > current_angle

        if desired_deg > current_angle:
            lo, hi = (0.0, max_dist) if going_up else (-max_dist, 0.0)
        else:
            lo, hi = (-max_dist, 0.0) if going_up else (0.0, max_dist)

        if not (min(angle_at(lo), angle_at(hi)) <= desired_deg
                <= max(angle_at(lo), angle_at(hi))):
            lo, hi = -lo, -hi

        for _ in range(50):
            mid = (lo + hi) * 0.5
            if (angle_at(mid) < desired_deg) == (angle_at(lo) < desired_deg):
                lo = mid
            else:
                hi = mid

        return c + bisector * ((lo + hi) * 0.5)
    return None


class MBF_OT_correct_angles(bpy.types.Operator):
    """
    Corrige los ángulos internos del plano activo para aproximarlos
    al ángulo objetivo. Útil para limpiar planos de planta irregulares
    antes de generar el edificio.
    """
    bl_idname  = "makibuildify.correct_angles"
    bl_label   = "Corregir ángulos del plano"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH'

    def execute(self, context):
        obj      = context.active_object
        s        = context.scene.mbf_settings
        # Convertir de radianes (subtype ANGLE) a grados
        target   = math.degrees(s.ac_target_angle)
        tol      = math.degrees(s.ac_tolerance)
        max_iter = s.ac_max_iters

        was_edit = (obj.mode == 'EDIT')
        if not was_edit:
            bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        bm.verts.ensure_lookup_table()

        corrected_total = 0
        low  = target - tol
        high = target + tol

        for iteration in range(max_iter):
            corrected_iter = 0
            for face in bm.faces:
                for v in face.verts:
                    angle = _internal_angle_at_vertex(v, face)
                    if angle is None or low <= angle <= high:
                        continue
                    desired = low if angle < low else high
                    new_pos = _corrected_position(v, face, desired)
                    if new_pos is not None:
                        v.co = new_pos
                        corrected_iter += 1

            corrected_total += corrected_iter
            if corrected_iter == 0:
                self.report({'INFO'},
                    f"Convergencia en iteración {iteration + 1}. "
                    f"Total correcciones: {corrected_total}")
                break
        else:
            self.report({'WARNING'},
                f"Máximo de iteraciones alcanzado. "
                f"Total correcciones: {corrected_total}")

        bmesh.update_edit_mesh(obj.data)

        if not was_edit:
            bpy.ops.object.mode_set(mode='OBJECT')

        return {'FINISHED'}


# ================================================================
#  PANEL
# ================================================================

class MBF_PT_main(bpy.types.Panel):
    bl_label       = "MakiBuildify"
    bl_idname      = "MBF_PT_main"
    bl_space_type  = "VIEW_3D"
    bl_region_type = "UI"
    bl_category    = "MakiBuildify"

    def draw(self, context):
        layout = self.layout
        s      = context.scene.mbf_settings
        obj    = context.active_object

        if obj is None or obj.type != 'MESH':
            layout.label(text="Selecciona un plano Mesh", icon='ERROR')
            return

        box = layout.box()
        if section_header(box, s, "open_config", "Configuración", 'PREFERENCES'):
            box.prop(s, "seed",     text="Semilla")
            box.prop(s, "n_floors", text="Pisos")
            box.prop(s, "storage",  text="Carpeta de almacenaje")

        layout.separator(factor=0.5)

        box = layout.box()
        if section_header(box, s, "open_fachada", "Módulos de fachada", 'MOD_BUILD'):
            for attr, lbl in [("base","Base"),("piso","Piso"),("cornisa","Cornisa")]:
                row = box.row(align=True)
                row.label(text=lbl)
                row.prop(getattr(s, attr), "collection", text="")

        layout.separator(factor=0.5)

        draw_corner_section(layout, s, "open_esq_ext",
                            "Esquinas Exteriores (salientes)",
                            s.esq_ext_base, s.esq_ext_piso, s.esq_ext_cornisa)
        layout.separator(factor=0.5)
        draw_corner_section(layout, s, "open_esq_int",
                            "Esquinas Interiores (entrantes)",
                            s.esq_int_base, s.esq_int_piso, s.esq_int_cornisa)

        layout.separator(factor=0.5)

        box = layout.box()
        if section_header(box, s, "open_pilar", "Pilares", 'SNAP_EDGE'):
            box.prop(s.pilar, "collection", text="Colección")
            box.prop(s.pilar, "density",    text="Densidad", slider=True)

        layout.separator(factor=0.5)

        box = layout.box()
        if section_header(box, s, "open_db", "Decoraciones — Base", 'FUND'):
            draw_deco_group(box, s.deco_base_a, "Grupo A")
            draw_deco_group(box, s.deco_base_b, "Grupo B")
            draw_deco_group(box, s.deco_base_c, "Grupo C")

        layout.separator(factor=0.5)

        box = layout.box()
        if section_header(box, s, "open_dp", "Decoraciones — Pisos", 'FUND'):
            draw_deco_group(box, s.deco_piso_a, "Grupo A")
            draw_deco_group(box, s.deco_piso_b, "Grupo B")
            draw_deco_group(box, s.deco_piso_c, "Grupo C")

        layout.separator(factor=0.5)

        box = layout.box()
        if section_header(box, s, "open_dc", "Decoraciones — Cornisa", 'FUND'):
            draw_deco_group(box, s.deco_cornisa_a, "Grupo A")
            draw_deco_group(box, s.deco_cornisa_b, "Grupo B")
            draw_deco_group(box, s.deco_cornisa_c, "Grupo C")

        layout.separator()

        # ── Corrector de ángulos ──
        box = layout.box()
        if section_header(box, s, "open_corrector", "Corrector de ángulos", 'DRIVER_ROTATIONAL_DIFFERENCE'):
            box.prop(s, "ac_target_angle", text="Ángulo objetivo")
            box.prop(s, "ac_tolerance",    text="Tolerancia ±")
            box.prop(s, "ac_max_iters",    text="Iteraciones máx.")
            box.operator("makibuildify.correct_angles",
                         text="Corregir ángulos", icon='CON_ROTLIKE')

        layout.separator()

        col = layout.column(align=True)
        col.scale_y = 1.5
        col.operator("makibuildify.generate", text="Generar edificio", icon='HOME')
        col.operator("makibuildify.clear",    text="Limpiar edificio", icon='TRASH')


# ================================================================
#  REGISTRO
# ================================================================

classes = [
    MBF_SlotProps,
    MBF_PilarProps,
    MBF_DecoGroupProps,
    MBF_SceneSettings,
    MBF_OT_generate,
    MBF_OT_clear,
    MBF_OT_correct_angles,
    MBF_PT_main,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mbf_settings = PointerProperty(type=MBF_SceneSettings)

def unregister():
    del bpy.types.Scene.mbf_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
