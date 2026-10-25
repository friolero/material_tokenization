import os
from pymatgen.io.cif import CifParser
from eval.cdvae.utils import get_crystal_array_list
from eval.utils import make_crystals_mp

def convert_cif(file_path, args, gen=None):
    structures_path = file_path+'/cifs'+str(args.num_samples)+'/'
    structures = []
    if os.path.exists(structures_path):
        args.logger.info("Have cif files already!")
        for i in range(args.num_samples):
            if os.path.exists(structures_path+str(i)+'.cif'):
                parser = CifParser(structures_path+str(i)+'.cif')
                structure = parser.get_structures()[0]
                structures.append(structure)
    else:
        os.makedirs(structures_path)
        gen_crys = make_crystals_mp(gen)
            
        for i in range(len(gen_crys)):
            x = gen_crys[i]
            if x is not None:
                structure = x.structure
                if structure is not None and x.valid:
                    structure.to(filename=structures_path+str(i)+'.cif')
                    structures.append(structure)
    args.logger.info("Num of generated materials: {:.0f}\t Successfully convert to CIF: {:.0f}".format(args.num_samples, len(structures)))
    return structures