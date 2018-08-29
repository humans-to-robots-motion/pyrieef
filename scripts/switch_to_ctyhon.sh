EXT1=pyx
EXT2=py
cd ../pyrieef/geometry/
mv differentiable_geometry.${EXT1} differentiable_geometry.${EXT2}
mv workspace.${EXT1} workspace.${EXT2}
cd -
cd ../pyrieef/motion/
mv cost_terms.${EXT1} cost_terms.${EXT2}
cd -
