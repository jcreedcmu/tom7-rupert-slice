#include "textured-mesh.h"

#include <cmath>
#include <cstdio>
#include <string_view>
#include <string>
#include <format>
#include <cstddef>

#include "ansi.h"
#include "base/stringprintf.h"
#include "image.h"
#include "util.h"
#include "yocto_matht.h"

void SaveAsOBJ(const TexturedMesh &tmesh, std::string_view filename_base) {
  std::string base = (std::string)filename_base;
  std::string obj_filename = base + ".obj";
  std::string mtl_filename = base + ".mtl";

  // Extract just the filename part for referencing within OBJ/MTL files.
  std::string mtl_ref_name = Util::FileOf(mtl_filename);

  // Apparently it's just a convention.
  std::string png_ref_name = std::format("{}.1001.png", base);
  for (int idx = 0; idx < tmesh.texture.NumTextures(); idx++) {
    const auto &[uu, vv] = tmesh.texture.UnpackUV(idx);
    int filenum = tmesh.texture.Filenum(uu, vv);
    tmesh.texture.GetTexture(uu, vv).Save(
        std::format("{}.{}.png", base, filenum));
  }

  // Material file.
  {
    std::string mtl_contents;
    // Define the material
    AppendFormat(&mtl_contents, "newmtl material0\n");
    // Ambient color (white)
    AppendFormat(&mtl_contents, "  Ka 1.0 1.0 1.0\n");
    // Diffuse color (white)
    AppendFormat(&mtl_contents, "  Kd 1.0 1.0 1.0\n");
    // Specular color (off)
    AppendFormat(&mtl_contents, "  Ks 0.0 0.0 0.0\n");
    // Specular exponent (low)
    AppendFormat(&mtl_contents, "  Ns 10.0\n");
    // Opacity (fully opaque)
    AppendFormat(&mtl_contents, "  d 1.0\n");
    // Illumination model (2 = highlight on)
    AppendFormat(&mtl_contents, "  illum 2\n");
    // Diffuse texture map
    AppendFormat(&mtl_contents, "  map_Kd {}\n", png_ref_name);

    Util::WriteFile(mtl_filename, mtl_contents);
  }

  // OBJ file.
  {
    std::string obj_contents;
    AppendFormat(&obj_contents, "mtllib {}\n\n", mtl_ref_name);

    // Vertex positions (v x y z)
    AppendFormat(&obj_contents, "# Vertex positions ({})\n",
                 tmesh.mesh.vertices.size());
    for (const auto &v : tmesh.mesh.vertices) {
      AppendFormat(&obj_contents, "v {:.17g} {:.17g} {:.17g}\n",
                   v.x, v.y, v.z);
    }
    obj_contents += "\n";

    // Texture coordinates (vt u v)
    AppendFormat(&obj_contents, "# Texture coordinates ({})\n",
                 tmesh.uvs.size() * 3);
    CHECK(tmesh.mesh.triangles.size() == tmesh.uvs.size())
      << "Triangle count (" << tmesh.mesh.triangles.size()
      << ") must match UV tuple count (" << tmesh.uvs.size() << ").";

    // Flip the v coordinate *within each udim tile*.
    auto FlipV = [](double y) {
      double tile = std::floor(y);
      double frac = y - tile;
      return tile + (1.0 - frac);
    };

    for (const auto &[uva, uvb, uvc] : tmesh.uvs) {
      AppendFormat(&obj_contents, "vt {:.17g} {:.17g}\n", uva.x, FlipV(uva.y));
      AppendFormat(&obj_contents, "vt {:.17g} {:.17g}\n", uvb.x, FlipV(uvb.y));
      AppendFormat(&obj_contents, "vt {:.17g} {:.17g}\n", uvc.x, FlipV(uvc.y));
    }

    // Use this same material for all faces.
    AppendFormat(&obj_contents, "\nusemtl material0\n");
    // Disable smoothing groups
    AppendFormat(&obj_contents, "s off\n\n");

    // Faces (f v1/vt1 v2/vt2 v3/vt3)
    AppendFormat(&obj_contents, "# Faces ({})\n",
                 tmesh.mesh.triangles.size());
    for (size_t i = 0; i < tmesh.mesh.triangles.size(); ++i) {
      const auto &[ia, ib, ic] = tmesh.mesh.triangles[i];

      // OBJ indices are 1-based.
      int va = ia + 1;
      int vb = ib + 1;
      int vc = ic + 1;

      // Texture coordinate indices are also 1-based.
      int vta = (int)(3 * i + 0) + 1;
      int vtb = (int)(3 * i + 1) + 1;
      int vtc = (int)(3 * i + 2) + 1;

      AppendFormat(&obj_contents, "f {}/{} {}/{} {}/{}\n",
                   va, vta, vb, vtb, vc, vtc);
    }

    Util::WriteFile(obj_filename, obj_contents);
  }

  printf("Wrote " AGREEN("%s") ", " AGREEN("%s") ", " AGREEN("%s")
         APURPLE("...") "\n",
         obj_filename.c_str(), mtl_filename.c_str(), png_ref_name.c_str());
}
