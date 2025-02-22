#!/bin/bash

make -j 20 tostl.exe || exit -1

./tostl.exe tetrahedron platonic-tetrahedron.stl
./tostl.exe cube platonic-cube.stl
./tostl.exe dodecahedron platonic-dodecahedron.stl
./tostl.exe icosahedron platonic-icosahedron.stl
./tostl.exe octahedron platonic-octahedron.stl

./tostl.exe truncatedtetrahedron archimedean-truncatedtetrahedron.stl
./tostl.exe cuboctahedron archimedean-cuboctahedron.stl
./tostl.exe truncatedcube archimedean-truncatedcube.stl
./tostl.exe truncatedoctahedron archimedean-truncatedoctahedron.stl
./tostl.exe rhombicuboctahedron archimedean-rhombicuboctahedron.stl
./tostl.exe truncatedcuboctahedron archimedean-truncatedcuboctahedron.stl
./tostl.exe snubcube archimedean-snubcube.stl
./tostl.exe icosidodecahedron archimedean-icosidodecahedron.stl
./tostl.exe truncateddodecahedron archimedean-truncateddodecahedron.stl
./tostl.exe truncatedicosahedron archimedean-truncatedicosahedron.stl
./tostl.exe rhombicosidodecahedron archimedean-rhombicosidodecahedron.stl
./tostl.exe truncatedicosidodecahedron archimedean-truncatedicosidodecahedron.stl
./tostl.exe snubdodecahedron archimedean-snubdodecahedron.stl

./tostl.exe triakistetrahedron catalan-triakistetrahedron.stl
./tostl.exe rhombicdodecahedron catalan-rhombicdodecahedron.stl
./tostl.exe triakisoctahedron catalan-triakisoctahedron.stl
./tostl.exe tetrakishexahedron catalan-tetrakishexahedron.stl
./tostl.exe deltoidalicositetrahedron catalan-deltoidalicositetrahedron.stl
./tostl.exe disdyakisdodecahedron catalan-disdyakisdodecahedron.stl
./tostl.exe deltoidalhexecontahedron catalan-deltoidalhexecontahedron.stl
./tostl.exe pentagonalicositetrahedron catalan-pentagonalicositetrahedron.stl
./tostl.exe rhombictriacontahedron catalan-rhombictriacontahedron.stl
./tostl.exe triakisicosahedron catalan-triakisicosahedron.stl
./tostl.exe pentakisdodecahedron catalan-pentakisdodecahedron.stl
./tostl.exe disdyakistriacontahedron catalan-disdyakistriacontahedron.stl
./tostl.exe pentagonalhexecontahedron catalan-pentagonalhexecontahedron.stl
