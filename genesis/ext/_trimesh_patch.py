import os
import re
import warnings
from collections import defaultdict, deque

import numpy as np

try:
    # `pip install pillow`
    # optional: used for textured meshes
    from PIL import Image
except BaseException as E:
    # if someone tries to use Image re-raise
    # the import error so they can debug easily
    from ..exceptions import ExceptionWrapper

    Image = ExceptionWrapper(E)

import trimesh
from trimesh import util
from trimesh.constants import log, tol
from trimesh.visual.color import to_float
from trimesh.visual.material import SimpleMaterial
from trimesh.visual.texture import TextureVisuals, unmerge_faces

import genesis as gs


# Silence trimesh mass_properties divide-by-zero on degenerate meshes.
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r"(divide by zero|invalid value) encountered in (divide|scalar multiply)",
    module=r"trimesh\..*",
)


def load_obj(
    file_obj,
    resolver=None,
    group_material=True,
    skip_materials=False,
    maintain_order=False,
    **kwargs,
):
    """
    Load a Wavefront OBJ file into kwargs for a trimesh.Scene
    object.

    Parameters
    --------------
    file_obj : file like object
      Contains OBJ data
    resolver : trimesh.visual.resolvers.Resolver
      Allow assets such as referenced textures and
      material files to be loaded
    group_material : bool
      Group faces that share the same material
      into the same mesh.
    skip_materials : bool
      Don't load any materials.
    maintain_order : bool or None
      Do not reorder faces or vertices which may result
      in visual artifacts.

    Returns
    -------------
    kwargs : dict
      Keyword arguments which can be loaded by
      trimesh.exchange.load.load_kwargs into a trimesh.Scene
    """
    # get text as bytes or string blob
    text = file_obj.read()
    # if text was bytes decode into string
    text = util.decode_text(text)

    # add leading and trailing newlines so we can use the
    # same logic even if they jump directly in to data lines
    text = "\n{}\n".format(text.strip().replace("\r\n", "\n"))

    # remove backslash continuation characters and merge them into the same
    # line
    text = text.replace("\\\n", "")

    # Load Materials
    materials = {}
    mtl_position = text.find("mtllib")
    if not skip_materials and mtl_position >= 0:
        # take the line of the material file after `mtllib`
        # which should be the file location of the .mtl file
        mtl_path = text[mtl_position + 6 : text.find("\n", mtl_position)].strip()

        # pyassimp can magically handle this
        if not os.path.exists(os.path.join(os.path.dirname(file_obj.name), mtl_path)):
            mtl_path = mtl_path.replace(" ", "_")

        try:
            # use the resolver to get the data
            material_kwargs = parse_mtl(resolver[mtl_path], resolver=resolver)
            # turn parsed kwargs into material objects
            materials = {k: SimpleMaterial(**v) for k, v in material_kwargs.items()}
        except (OSError, TypeError):
            # usually the resolver couldn't find the asset
            log.debug(f"unable to load materials from: {mtl_path}")
        except BaseException:
            # something else happened so log a warning
            log.debug(f"unable to load materials from: {mtl_path}", exc_info=True)

    # extract vertices from raw text
    v, vn, vt, vc = _parse_vertices(text=text)

    # get relevant chunks that have face data
    # in the form of (material, object, chunk)
    face_tuples = _preprocess_faces(text=text)

    # combine chunks that have the same material
    # some meshes end up with a LOT of components
    # and will be much slower if you don't do this
    if group_material:
        face_tuples = _group_by_material(face_tuples)

    # no faces but points given
    # return point cloud
    if not len(face_tuples) and v is not None:
        pc = {"vertices": v}
        if vn is not None:
            pc["vertex_normals"] = vn
        if vc is not None:
            pc["vertex_colors"] = vc
        return pc

    # Load Faces
    # now we have clean- ish faces grouped by material and object
    # so now we have to turn them into numpy arrays and kwargs
    # for trimesh mesh and scene objects
    geometry = {}
    while len(face_tuples) > 0:
        # consume the next chunk of text
        material, current_object, chunk = face_tuples.pop()
        # do wangling in string form
        # we need to only take the face line before a newline
        # using builtin functions in a list comprehension
        # is pretty fast relative to other options
        # this operation is the only one that is O(len(faces))
        # slower due to the tight-loop conditional:
        # face_lines = [i[:i.find('\n')]
        #              for i in chunk.split('\nf ')[1:]
        #              if i.rfind('\n') >0]
        # maxsplit=1 means that it can stop working
        # after it finds the first newline
        # passed as arg as it's not a kwarg in python2
        face_lines = [i.split("\n", 1)[0].strip() for i in re.split("^f", chunk, flags=re.MULTILINE)[1:]]

        # check every face for mixed tri-quad-ngon
        columns = len(face_lines[0].replace("/", " ").split())
        flat_array = all(columns == len(f.replace("/", " ").split()) for f in face_lines)

        # make sure we have the right number of values for vectorized
        if flat_array:
            # the fastest way to get to a numpy array
            # processes the whole string at once into a 1D array
            array = np.fromstring(" ".join(face_lines).replace("/", " "), sep=" ", dtype=np.int64)
            # also wavefront is 1-indexed (vs 0-indexed) so offset
            # only applies to positive indices
            array[array > 0] -= 1

            # everything is a nice 2D array
            faces, faces_tex, faces_norm = _parse_faces_vectorized(
                array=array, columns=columns, sample_line=face_lines[0]
            )
        else:
            # if we had something annoying like mixed in quads
            # or faces that differ per-line we have to loop
            # i.e. something like:
            #  '31407 31406 31408',
            #  '32303/2469 32304/2469 32305/2469',
            log.debug("faces have mixed data: using slow fallback!")
            faces, faces_tex, faces_norm = _parse_faces_fallback(face_lines)

        if group_material:
            name = material
        else:
            name = current_object

        # ensure the name is always unique
        name = util.unique_name(name, geometry)

        # try to get usable texture
        mesh = kwargs.copy()
        if faces_tex is not None:
            # convert faces referencing vertices and
            # faces referencing vertex texture to new faces
            # where each face
            if faces_norm is not None and len(faces_norm) == len(faces):
                new_faces, mask_v, mask_vt, mask_vn = unmerge_faces(
                    faces, faces_tex, faces_norm, maintain_faces=maintain_order
                )
            else:
                mask_vn = None
                # no face normals but face texturre
                new_faces, mask_v, mask_vt = unmerge_faces(faces, faces_tex, maintain_faces=maintain_order)

            if tol.strict:
                # we should NOT have messed up the faces
                # note: this is EXTREMELY slow due to all the
                # float comparisons so only run this in unit tests
                assert np.allclose(v[faces], v[mask_v][new_faces])
                # faces should all be in bounds of vertives
                assert new_faces.max() < len(v[mask_v])
            try:
                # survive index errors as sometimes we
                # want materials without UV coordinates
                uv = vt[mask_vt]
            except BaseException:
                log.debug("index failed on UV coordinates, skipping!")
                uv = None

            # mask vertices and use new faces
            mesh.update({"vertices": v[mask_v].copy(), "faces": new_faces})

        else:
            # otherwise just use unmasked vertices
            uv = None
            # check to make sure indexes are in bounds
            if tol.strict:
                assert faces.max() < len(v)
            if vn is not None and np.shape(faces_norm) == faces.shape:
                # do the crazy unmerging logic for split indices
                new_faces, mask_v, mask_vn = unmerge_faces(faces, faces_norm, maintain_faces=maintain_order)
            else:
                # generate the mask so we only include
                # referenced vertices in every new mesh
                mask_v = np.zeros(len(v), dtype=bool)
                mask_v[faces] = True

                # reconstruct the faces with the new vertex indices
                inverse = np.zeros(len(v), dtype=np.int64)
                inverse[mask_v] = np.arange(mask_v.sum())
                new_faces = inverse[faces]
                # no normals
                mask_vn = None

            # start with vertices and faces
            mesh.update({"faces": new_faces, "vertices": v[mask_v].copy()})

        # if colors and normals are OK save them
        if vc is not None:
            try:
                # may fail on a malformed color mask
                mesh["vertex_colors"] = vc[mask_v]
            except BaseException:
                log.debug("failed to load vertex_colors", exc_info=True)
        if mask_vn is not None:
            try:
                # may fail on a malformed mask
                normals = vn[mask_vn]
                if normals.shape != mesh["vertices"].shape:
                    raise ValueError(
                        "incorrect normals {} != {}".format(str(normals.shape), str(mesh["vertices"].shape))
                    )
                mesh["vertex_normals"] = normals
            except BaseException:
                log.debug("failed to load vertex_normals", exc_info=True)

        visual = None
        if material in materials:
            # use the material with the UV coordinates
            visual = TextureVisuals(uv=uv, material=materials[material])
        elif uv is not None and len(uv) == len(mesh["vertices"]):
            # create a texture with an empty materials
            visual = TextureVisuals(uv=uv)
        elif material is not None:
            # case where material is specified but not available
            log.debug(f"specified material ({material})  not loaded!")
        # assign the visual
        mesh["visual"] = visual
        # store geometry by name
        geometry[name] = mesh

    if len(geometry) == 1:
        # TODO : should this be removed to always return a scene?
        return next(iter(geometry.values()))

    # add an identity transform for every geometry
    graph = [{"geometry": k, "frame_to": k} for k in geometry.keys()]

    # convert to scene kwargs
    result = {"geometry": geometry, "graph": graph}

    return result


def parse_mtl(mtl, resolver=None):
    """
    Parse a loaded MTL file.

    Parameters
    -------------
    mtl : str or bytes
      Data from an MTL file
    resolver : trimesh.Resolver
      Fetch assets by name from file system, web, or other

    Returns
    ------------
    mtllibs : list of dict
      Each dict has keys: newmtl, map_Kd, Kd
    """
    # decode bytes into string if necessary
    mtl = util.decode_text(mtl)

    # current material
    material = None
    # materials referenced by name
    materials = {}
    # use universal newline splitting
    lines = str.splitlines(str(mtl).strip())

    # remap OBJ property names to kwargs for SimpleMaterial
    mapped = {"kd": "diffuse", "ka": "ambient", "ks": "specular", "ns": "glossiness"}

    for line in lines:
        # split by white space
        split = line.strip().split()
        # needs to be at least two values
        if len(split) <= 1:
            continue
        # the first value is the parameter name
        key = split[0].lower()
        # start a new material
        if key == "newmtl":
            # material name extracted from line like:
            # newmtl material_0
            if material is not None:
                # save the old material by old name and remove key
                materials[material.pop("newmtl")] = material
            # start a fresh new material
            material = {"newmtl": " ".join(split[1:])}

        elif key == "map_kd":
            # represents the file name of the texture image
            index = line.lower().index("map_kd") + 6
            file_name = line[index:].strip()

            # sometimes we have weird format like: map_Kd -s 6.2 11.8 1.0 OAKQRTRT.TGA
            file_name = file_name.split()[-1]

            try:
                file_data = resolver.get(file_name)
                # load the bytes into a PIL image
                # an image file name
                material["image"] = Image.open(util.wrap_as_stream(file_data))
            except BaseException:
                log.debug("failed to load image", exc_info=True)

        elif key in mapped.keys():
            try:
                # diffuse, ambient, and specular float RGB
                value = [float(x) for x in split[1:]]
                # if there is only one value return that
                if len(value) == 1:
                    value = value[0]
                if material is not None:
                    # store the key by mapped value
                    material[mapped[key]] = value
                    # also store key by OBJ name
                    material[key] = value
            except BaseException:
                log.debug("failed to convert color!", exc_info=True)
        # pass everything as kwargs to material constructor
        elif material is not None:
            # save any other unspecified keys
            material[key] = split[1:]
    # reached EOF so save any existing materials
    if material:
        materials[material.pop("newmtl")] = material

    return materials


def _parse_faces_vectorized(array, columns, sample_line):
    """
    Parse loaded homogeneous (tri/quad) face data in a
    vectorized manner.

    Parameters
    ------------
    array : (n,) int
      Indices in order
    columns : int
      Number of columns in the file
    sample_line : str
      A single line so we can assess the ordering

    Returns
    --------------
    faces : (n, d) int
      Faces in space
    faces_tex : (n, d) int or None
      Texture for each vertex in face
    faces_norm : (n, d) int or None
      Normal index for each vertex in face
    """
    # reshape to columns
    array = array.reshape((-1, columns))
    # how many elements are in the first line of faces
    # i.e '13/1/13 14/1/14 2/1/2 1/2/1' is 4
    group_count = len(sample_line.strip().split())
    # how many elements are there for each vertex reference
    # i.e. '12/1/13' is 3
    per_ref = int(columns / group_count)
    # create an index mask we can use to slice vertex references
    index = np.arange(group_count) * per_ref
    # slice the faces out of the blob array
    faces = array[:, index]

    # TODO: probably need to support 8 and 12 columns for quads
    # or do something more general
    faces_tex, faces_norm = None, None
    if columns == 6:
        # if we have two values per vertex the second
        # one is index of texture coordinate (`vt`)
        # count how many delimiters are in the first face line
        # to see if our second value is texture or normals
        # do splitting to clip off leading/trailing slashes
        count = "".join(i.strip("/") for i in sample_line.split()).count("/")
        if count == columns:
            # case where each face line looks like:
            # ' 75//139 76//141 77//141'
            # which is vertex/nothing/normal
            faces_norm = array[:, index + 1]
        elif count == int(columns / 2):
            # case where each face line looks like:
            # '75/139 76/141 77/141'
            # which is vertex/texture
            faces_tex = array[:, index + 1]
        else:
            log.debug(f"face lines are weird: {sample_line}")
    elif columns == 8:
        gs.raise_exception("Not supported yet. Contact us if you need this.")
    elif columns == 12:
        # if we have three values per vertex
        # second value is always texture
        faces_tex = array[:, index + 1]
        # third value is reference to vertex normal (`vn`)
        faces_norm = array[:, index + 2]
    elif columns == 9:
        # if we have three values per vertex
        # second value is always texture
        faces_tex = array[:, index + 1]
        # third value is reference to vertex normal (`vn`)
        faces_norm = array[:, index + 2]
    return faces, faces_tex, faces_norm


def _parse_faces_fallback(lines):
    """
    Use a slow but more flexible looping method to process
    face lines as a fallback option to faster vectorized methods.

    Parameters
    -------------
    lines : (n,) str
      List of lines with face information

    Returns
    -------------
    faces : (m, 3) int
      Clean numpy array of face triangles
    """

    # collect vertex, texture, and vertex normal indexes
    v, vt, vn = [], [], []

    # loop through every line starting with a face
    for line in lines:
        # remove leading newlines then
        # take first bit before newline then split by whitespace
        split = line.strip().split("\n")[0].split()
        # split into: ['76/558/76', '498/265/498', '456/267/456']
        len_split = len(split)
        if len_split == 3:
            pass
        elif len_split == 4:
            # triangulate quad face
            split = [split[0], split[1], split[2], split[2], split[3], split[0]]
        elif len_split > 4:
            # triangulate polygon as a triangles fan
            collect = []
            # we need a flat list so append inside
            # a list comprehension
            collect_append = collect.append
            [
                [
                    collect_append(split[0]),
                    collect_append(split[i + 1]),
                    collect_append(split[i + 2]),
                ]
                for i in range(len(split) - 2)
            ]
            split = collect
        else:
            log.debug(f"face needs more values 3>{len(split)} skipping!")
            continue

        # f is like: '76/558/76'
        for f in split:
            # vertex, vertex texture, vertex normal
            split = f.split("/")
            # we always have a vertex reference
            v.append(int(split[0]))

            # faster to try/except than check in loop
            try:
                vt.append(int(split[1]))
            except BaseException:
                pass
            try:
                # vertex normal is the third index
                vn.append(int(split[2]))
            except BaseException:
                pass

    # shape into triangles and switch to 0-indexed
    # 0-indexing only applies to positive indices
    faces = np.array(v, dtype=np.int64).reshape((-1, 3))
    faces[faces > 0] -= 1
    faces_tex, normals = None, None
    if len(vt) == len(v):
        faces_tex = np.array(vt, dtype=np.int64).reshape((-1, 3))
        faces_tex[faces_tex > 0] -= 1
    if len(vn) == len(v):
        normals = np.array(vn, dtype=np.int64).reshape((-1, 3))
        normals[normals > 0] -= 1

    return faces, faces_tex, normals


def _parse_vertices(text):
    """
    Parse raw OBJ text into vertices, vertex normals,
    vertex colors, and vertex textures.

    Parameters
    -------------
    text : str
      Full text of an OBJ file

    Returns
    -------------
    v : (n, 3) float
      Vertices in space
    vn : (m, 3) float or None
      Vertex normals
    vt : (p, 2) float or None
      Vertex texture coordinates
    vc : (n, 3) float or None
      Per-vertex color
    """

    # the first position of a vertex in the text blob
    # we only really need to search from the start of the file
    # up to the location of out our first vertex but we
    # are going to use this check for "do we have texture"
    # determination later so search the whole stupid file
    starts = {k: text.find(f"\n{k} ") for k in ["v", "vt", "vn"]}

    # no valid values so exit early
    if not any(v >= 0 for v in starts.values()):
        return None, None, None, None

    # find the last position of each valid value
    ends = {k: text.find("\n", text.rfind(f"\n{k} ") + 2 + len(k)) for k, v in starts.items() if v >= 0}

    # take the first and last position of any vertex property
    start = min(s for s in starts.values() if s >= 0)
    end = max(e for e in ends.values() if e >= 0)
    # get the chunk of test that contains vertex data
    chunk = text[start:end].replace("+e", "e").replace("-e", "e")

    # get the clean-ish data from the file as python lists
    data = {k: [i.split("\n", 1)[0] for i in chunk.split(f"\n{k} ")[1:]] for k, v in starts.items() if v >= 0}

    # count the number of data values per row on a sample row
    per_row = {k: len(v[0].split()) for k, v in data.items()}

    # convert data values into numpy arrays
    result = defaultdict(lambda: None)
    for k, value in data.items():
        # use joining and fromstring to get as numpy array
        array = np.fromstring(" ".join(value), sep=" ", dtype=np.float64)
        # what should our shape be
        shape = (len(value), per_row[k])
        # check shape of flat data
        if len(array) == np.prod(shape):
            # we have a nice 2D array
            result[k] = array.reshape(shape)
        else:
            # try to recover with a slightly more expensive loop
            count = per_row[k]
            try:
                # try to get result through reshaping
                result[k] = np.fromstring(
                    " ".join(i.split()[:count] for i in value), sep=" ", dtype=np.float64
                ).reshape(shape)
            except BaseException:
                pass

    # vertices
    v = result["v"]
    # vertex colors are stored next to vertices
    vc = None
    if v is not None and v.shape[1] >= 6:
        # vertex colors are stored after vertices
        v, vc = v[:, :3], v[:, 3:6]
    elif v is not None and v.shape[1] > 3:
        # we got a lot of something unknowable
        v = v[:, :3]

    # vertex texture or None
    vt = result["vt"]
    if vt is not None:
        # sometimes UV coordinates come in as UVW
        vt = vt[:, :2]
    # vertex normals or None
    vn = result["vn"]

    # check will generally only be run in unit tests
    # so we are allowed to do things that are slow
    if tol.strict:
        # check to make sure our subsetting
        # didn't miss any vertices or data
        assert len(v) == text.count("\nv ")
        # make sure optional data matches file too
        if vn is not None:
            assert len(vn) == text.count("\nvn ")
        if vt is not None:
            assert len(vt) == text.count("\nvt ")

    return v, vn, vt, vc


def _group_by_material(face_tuples):
    """
    For chunks of faces split by material group
    the chunks that share the same material.

    Parameters
    ------------
    face_tuples : (n,) list of (material, obj, chunk)
      The data containing faces

    Returns
    ------------
    grouped : (m,) list of (material, obj, chunk)
      Grouped by material
    """

    # store the chunks grouped by material
    grouped = defaultdict(lambda: ["", "", []])
    # loop through existring
    for material, obj, chunk in face_tuples:
        grouped[material][0] = material
        grouped[material][1] = obj
        # don't do a million string concatenations in loop
        grouped[material][2].append(chunk)
    # go back and do a join to make a single string
    for k in grouped.keys():
        grouped[k][2] = "\n".join(grouped[k][2])
    # return as list
    return list(grouped.values())


def _preprocess_faces(text):
    """
    Pre-Process Face Text

    Rather than looking at each line in a loop we're
    going to split lines by directives which indicate
    a new mesh, specifically 'usemtl' and 'o' keys
    search for materials, objects, faces, or groups

    Parameters
    ------------
    text : str
      Raw file

    Returns
    ------------
    triple : (n, 3) tuple
      Tuples of (material, object, data-chunk)
    """
    # see which chunk is relevant
    starters = ["\nusemtl ", "\no ", "\nf ", "\ng ", "\ns "]
    f_start = len(text)
    # first index of material, object, face, group, or smoother
    for st in starters:
        search = text.find(st, 0, f_start)
        # if not contained find will return -1
        if search < 0:
            continue
        # subtract the length of the key from the position
        # to make sure it's included in the slice of text
        if search < f_start:
            f_start = search
    # index in blob of the newline after the last face
    f_end = text.find("\n", text.rfind("\nf ") + 3)
    # get the chunk of the file that has face information
    if f_end >= 0:
        # clip to the newline after the last face
        f_chunk = text[f_start:f_end]
    else:
        # no newline after last face
        f_chunk = text[f_start:]

    if tol.strict:
        # check to make sure our subsetting didn't miss any faces
        assert f_chunk.count("\nf ") == text.count("\nf ")

    # two things cause new meshes to be created:
    # objects and materials
    # re.finditer was faster than find in a loop
    # find the index of every material change
    idx_mtl = np.array([m.start(0) for m in re.finditer("usemtl ", f_chunk)], dtype=int)
    # find the index of every new object
    idx_obj = np.array([m.start(0) for m in re.finditer("\no ", f_chunk)], dtype=int)

    # find all the indexes where we want to split
    splits = np.unique(np.concatenate(([0, len(f_chunk)], idx_mtl, idx_obj)))

    # track the current material and object ID
    current_obj = None
    current_mtl = None
    # store (material, object, face lines)
    face_tuples = []

    for start, end in zip(splits[:-1], splits[1:]):
        # ensure there's always a trailing newline
        chunk = f_chunk[start:end].strip() + "\n"
        if chunk.startswith("o "):
            current_obj, chunk = chunk.split("\n", 1)
            current_obj = current_obj[2:].strip()
        elif chunk.startswith("usemtl"):
            current_mtl, chunk = chunk.split("\n", 1)
            current_mtl = current_mtl[6:].strip()
        # Discard the g tag line in the list of faces
        elif chunk.startswith("g "):
            _, chunk = chunk.split("\n", 1)
        # If we have an f at the beginning of a line
        # then add it to the list of faces chunks
        if chunk.startswith("f ") or "\nf" in chunk:
            face_tuples.append((current_mtl, current_obj, chunk))
    return face_tuples


def export_obj(
    mesh,
    include_normals=None,
    include_color=True,
    include_texture=True,
    return_texture=False,
    write_texture=True,
    resolver=None,
    digits=8,
    mtl_name=None,
    header="https://github.com/mikedh/trimesh",
):
    """
    Export a mesh as a Wavefront OBJ file.
    TODO: scenes with textured meshes

    Parameters
    -----------
    mesh : trimesh.Trimesh
      Mesh to be exported
    include_normals : Optional[bool]
      Include vertex normals in export. If None
      will only be included if vertex normals are in cache.
    include_color : bool
      Include vertex color in export
    include_texture : bool
      Include `vt` texture in file text
    return_texture : bool
      If True, return a dict with texture files
    write_texture : bool
      If True and a writable resolver is passed
      write the referenced texture files with resolver
    resolver : None or trimesh.resolvers.Resolver
      Resolver which can write referenced text objects
    digits : int
      Number of digits to include for floating point
    mtl_name : None or str
      If passed, the file name of the MTL file.
    header : str or None
      Header string for top of file or None for no header.

    Returns
    -----------
    export : str
      OBJ format output
    texture : dict
      Contains files that need to be saved in the same
      directory as the exported mesh: {file name : bytes}
    """
    # store the multiple options for formatting
    # vertex indexes for faces
    face_formats = {
        ("v",): "{}",
        ("v", "vn"): "{}//{}",
        ("v", "vt"): "{}/{}",
        ("v", "vn", "vt"): "{}/{}/{}",
    }

    # check the input
    if util.is_instance_named(mesh, "Trimesh"):
        meshes = [mesh]
    elif util.is_instance_named(mesh, "Scene"):
        meshes = mesh.dump()
    elif util.is_instance_named(mesh, "PointCloud"):
        meshes = [mesh]
    else:
        raise ValueError("must be Trimesh or Scene!")

    # collect lines to export
    objects = deque([])
    # keep track of the number of each export element
    counts = {"v": 0, "vn": 0, "vt": 0}
    # collect materials as we go
    materials = {}
    materials_name = set()

    for current in meshes:
        # we are going to reference face_formats with this
        face_type = ["v"]
        # OBJ includes vertex color as RGB elements on the same line
        if include_color and current.visual.kind in ["vertex", "face"] and len(current.visual.vertex_colors):
            # create a stacked blob with position and color
            v_blob = np.column_stack((current.vertices, to_float(current.visual.vertex_colors[:, :3])))
        else:
            # otherwise just export vertices
            v_blob = current.vertices

            # add the first vertex key and convert the array
        # add the vertices
        export = deque(["v " + util.array_to_string(v_blob, col_delim=" ", row_delim="\nv ", digits=digits)])

        # if include_normals is None then
        # only include if they're already stored
        if include_normals is None:
            include_normals = "vertex_normals" in current._cache.cache

        if include_normals:
            try:
                converted = util.array_to_string(
                    current.vertex_normals,
                    col_delim=" ",
                    row_delim="\nvn ",
                    digits=digits,
                )
                # if vertex normals are stored in cache export them
                face_type.append("vn")
                export.append("vn " + converted)
            except BaseException:
                log.debug("failed to convert vertex normals", exc_info=True)

        # collect materials into a dict
        if include_texture and hasattr(current.visual, "uv"):
            try:
                # get a SimpleMaterial
                material = current.visual.material
                if hasattr(material, "to_simple"):
                    material = material.to_simple()

                # hash the material to avoid duplicates
                hashed = hash(material)
                if hashed not in materials:
                    # get a unique name for the material
                    name = util.unique_name(material.name, materials_name)
                    # add the name to our collection
                    materials_name.add(name)
                    # convert material to an OBJ MTL
                    materials[hashed] = material.to_obj(name=name)

                # get the name of the current material as-stored
                tex_name = materials[hashed][1]

                # export the UV coordinates
                if len(np.shape(getattr(current.visual, "uv", None))) == 2:
                    converted = util.array_to_string(current.visual.uv, col_delim=" ", row_delim="\nvt ", digits=digits)
                    # if vertex texture exists and is the right shape
                    face_type.append("vt")
                    # add the uv coordinates
                    export.append("vt " + converted)
                # add the directive to use the exported material
                export.appendleft(f"usemtl {tex_name}")
            except BaseException:
                log.debug("failed to convert UV coordinates", exc_info=True)

        # the format for a single vertex reference of a face
        face_format = face_formats[tuple(face_type)]
        # add the exported faces to the export if available
        if hasattr(current, "faces"):
            export.append(
                "f "
                + util.array_to_string(
                    current.faces + 1 + counts["v"],
                    col_delim=" ",
                    row_delim="\nf ",
                    value_format=face_format,
                )
            )
        # offset our vertex position
        counts["v"] += len(current.vertices)

        # add object name if found in metadata
        if "name" in current.metadata:
            export.appendleft("\no {}".format(current.metadata["name"]))
        # add this object
        objects.append("\n".join(export))

    # collect files like images to write
    mtl_data = {}
    # combine materials
    if len(materials) > 0:
        # collect text for a single mtllib file
        mtl_lib = []
        # now loop through: keys are garbage hash
        # values are (data, name)
        for data, _ in materials.values():
            for file_name, file_data in data.items():
                if file_name.lower().endswith(".mtl"):
                    # collect mtl lines into single file
                    mtl_lib.append(file_data)
                elif file_name not in mtl_data:
                    # things like images
                    mtl_data[file_name] = file_data
                else:
                    log.warning(f"not writing {file_name}")

        if mtl_name is None:
            # if no name passed set a default
            mtl_name = "material.mtl"

        # prepend a header to the MTL text if requested
        if header is not None:
            prepend = f"# {header}\n\n".encode()
        else:
            prepend = b""

        # save the material data
        mtl_data[mtl_name] = prepend + b"\n\n".join(mtl_lib)
        # add the reference to the MTL file
        objects.appendleft(f"mtllib {mtl_name}")

    if header is not None:
        # add a created-with header to the top of the file
        objects.appendleft(f"# {header}")

    # add a trailing newline
    objects.append("\n")

    # combine elements into a single string
    text = "\n".join(objects)

    # if we have a resolver and have asked to write texture
    if write_texture and resolver is not None and len(materials) > 0:
        # not all resolvers have a write method
        [resolver.write(k, v) for k, v in mtl_data.items()]

    # if we exported texture it changes returned values
    if return_texture:
        return text, mtl_data

    return text


trimesh.exchange.obj.load_obj = load_obj
trimesh.exchange.load.mesh_loaders.update({"obj": load_obj})
