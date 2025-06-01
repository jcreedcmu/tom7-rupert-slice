#!/bin/bash

make -j -C .. tomov.exe || exit -1

../tomov.exe tetrahedron rotating-tetrahedron.mov
../tomov.exe cube rotating-cube.mov
../tomov.exe dodecahedron rotating-dodecahedron.mov
../tomov.exe icosahedron rotating-icosahedron.mov
../tomov.exe octahedron rotating-octahedron.mov
../tomov.exe truncatedtetrahedron rotating-truncatedtetrahedron.mov
../tomov.exe cuboctahedron rotating-cuboctahedron.mov
../tomov.exe truncatedcube rotating-truncatedcube.mov
../tomov.exe truncatedoctahedron rotating-truncatedoctahedron.mov
../tomov.exe rhombicuboctahedron rotating-rhombicuboctahedron.mov
../tomov.exe truncatedcuboctahedron rotating-truncatedcuboctahedron.mov
../tomov.exe snubcube rotating-snubcube.mov
../tomov.exe icosidodecahedron rotating-icosidodecahedron.mov
../tomov.exe truncateddodecahedron rotating-truncateddodecahedron.mov
../tomov.exe truncatedicosahedron rotating-truncatedicosahedron.mov
../tomov.exe rhombicosidodecahedron rotating-rhombicosidodecahedron.mov
../tomov.exe truncatedicosidodecahedron rotating-truncatedicosidodecahedron.mov
../tomov.exe snubdodecahedron rotating-snubdodecahedron.mov
../tomov.exe triakistetrahedron rotating-triakistetrahedron.mov
../tomov.exe rhombicdodecahedron rotating-rhombicdodecahedron.mov
../tomov.exe triakisoctahedron rotating-triakisoctahedron.mov
../tomov.exe tetrakishexahedron rotating-tetrakishexahedron.mov
../tomov.exe deltoidalicositetrahedron rotating-deltoidalicositetrahedron.mov
../tomov.exe disdyakisdodecahedron rotating-disdyakisdodecahedron.mov
../tomov.exe deltoidalhexecontahedron rotating-deltoidalhexecontahedron.mov
../tomov.exe pentagonalicositetrahedron rotating-pentagonalicositetrahedron.mov
../tomov.exe rhombictriacontahedron rotating-rhombictriacontahedron.mov
../tomov.exe triakisicosahedron rotating-triakisicosahedron.mov
../tomov.exe pentakisdodecahedron rotating-pentakisdodecahedron.mov
../tomov.exe disdyakistriacontahedron rotating-disdyakistriacontahedron.mov
../tomov.exe pentagonalhexecontahedron rotating-pentagonalhexecontahedron.mov

mv rotating-*.mov /d/video/ruperts/rotating-shapes/
