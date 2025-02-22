#!/bin/bash

make -j 20 soltostl.exe || exit -1

./soltostl.exe tetrahedron &
./soltostl.exe cube &
./soltostl.exe dodecahedron &
./soltostl.exe icosahedron &
./soltostl.exe octahedron &
wait

./soltostl.exe truncatedtetrahedron &
./soltostl.exe cuboctahedron &
./soltostl.exe truncatedcube &
./soltostl.exe truncatedoctahedron &
./soltostl.exe rhombicuboctahedron &
./soltostl.exe truncatedcuboctahedron &
wait

# unsolved
# ./soltostl.exe snubcube
./soltostl.exe icosidodecahedron &
./soltostl.exe truncateddodecahedron &
./soltostl.exe truncatedicosahedron &
# unsolved
# ./soltostl.exe rhombicosidodecahedron
./soltostl.exe truncatedicosidodecahedron &
# unsolved
# ./soltostl.exe snubdodecahedron
wait

./soltostl.exe triakistetrahedron &
./soltostl.exe rhombicdodecahedron &
./soltostl.exe triakisoctahedron &
./soltostl.exe tetrakishexahedron &
./soltostl.exe deltoidalicositetrahedron &
./soltostl.exe disdyakisdodecahedron &
wait

# unsolved
# ./soltostl.exe deltoidalhexecontahedron
./soltostl.exe pentagonalicositetrahedron &
./soltostl.exe rhombictriacontahedron &
./soltostl.exe triakisicosahedron &
./soltostl.exe pentakisdodecahedron &
./soltostl.exe disdyakistriacontahedron &
# unsolved
# ./soltostl.exe pentagonalhexecontahedron

wait
