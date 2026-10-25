import time
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from types import SimpleNamespace
import os
from main import load_model
from eval.cdvae.utils import get_ckpt, get_index2path
from utils import get_scaler

def process_one(crystal, matrix_scaler, index2type):
    ul = crystal[:3, :3]
    bl = crystal[3:, :3]
    br = crystal[3:, 3:]

    none_type = br.shape[-1]-1
    atom_types = br.argmax(-1)

    real_type, real_loc = [], []
    for i in range(len(atom_types)):
        if atom_types[i]!=none_type:
            try:
                real_type.append(index2type[atom_types[i].item()])
                loc = bl[i]
                loc[loc<0] = 0.
                real_loc.append(loc)
            except KeyError:
                pass
    atomic_number = torch.FloatTensor(real_type)
    locs = torch.stack(real_loc, 0)
    num_atoms = len(atomic_number)
    matrix = ul

    inverse_matrix = matrix_scaler.inverse_transform(matrix)
    real_matrix = inverse_matrix * num_atoms**(1/3)
    return {'atomic_number': atomic_number, 'location': locs, 'num_atoms': num_atoms, 'matrix': real_matrix}

def deal_result(x, matrix_scaler, index2type):
    batch_frac_coords, batch_num_atoms, batch_atom_types, batch_matrices = [], [], [], []
    all_results = []
    for crystal in tqdm(x, desc='discreting'):
        all_results.append(process_one(crystal, matrix_scaler, index2type))
    atomic_number, location, num_atom, matrix = [], [], [], []
    for re in all_results:
        if re != None:
            atomic_number.append(re['atomic_number'])
            location.append(re['location'])
            num_atom.append(torch.LongTensor([re['num_atoms']]))
            matrix.append(re['matrix'])
    atomic_number, location, num_atom, matrix = \
        torch.cat(atomic_number, 0), torch.cat(location, 0), torch.cat(num_atom, 0), torch.stack(matrix, 0)
    samples = {'atom_types': atomic_number, 'frac_coords': location, 'num_atoms':num_atom, 'matrices':matrix}
    batch_frac_coords.append(samples['frac_coords'].detach().cpu())
    batch_num_atoms.append(samples['num_atoms'].detach().cpu())
    batch_atom_types.append(samples['atom_types'].detach().cpu())
    batch_matrices.append(samples['matrices'].detach().cpu())

    batch_frac_coords, batch_num_atoms, batch_atom_types, batch_matrices = torch.stack(batch_frac_coords, dim=0), \
        torch.stack(batch_num_atoms, dim=0), torch.stack(batch_atom_types, dim=0), torch.stack(batch_matrices, dim=0)
    return batch_frac_coords, batch_num_atoms, batch_atom_types, batch_matrices

def generation(model, ld_kwargs, num_samples, conds, batch_size=1000):
    num_batches_to_sample = np.ceil(num_samples / batch_size).astype(np.int32)
    for z_idx in range(num_batches_to_sample):
        print(z_idx)
        with torch.no_grad():
            curr_cond = conds[z_idx*batch_size: min((z_idx+1)*batch_size, num_samples)]
            samples = model.reverse(min(batch_size, num_samples-z_idx*batch_size), curr_cond)
        torch.save(samples, ld_kwargs.model_path+'/'+'eval_gen_'+str(z_idx)+'.pt')
    frac_coords = []
    num_atoms = []
    atom_types = []
    matrices = []
    
    matrix_scaler = get_scaler(ld_kwargs)
    index2type = get_index2path(ld_kwargs)
    for z_idx in range(num_batches_to_sample):
        print(z_idx)
        batch_frac_coords, batch_num_atoms, batch_atom_types, batch_matrices = [], [], [], []

        gen_data = torch.load(ld_kwargs.model_path+'/'+'eval_gen_'+str(z_idx)+'.pt').detach().cpu()
        x = gen_data
        
        batch_frac_coords, batch_num_atoms, batch_atom_types, batch_matrices = deal_result(x, matrix_scaler, index2type)

        frac_coords.append(batch_frac_coords)
        num_atoms.append(batch_num_atoms)
        atom_types.append(batch_atom_types)
        matrices.append(batch_matrices)

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    matrices = torch.cat(matrices, dim=1)
    return (frac_coords, num_atoms, atom_types, matrices)

def generate_material(args, config, conds):
    logger = args.logger
    model_path = Path(args.model_path)
    ckpt = get_ckpt(os.path.join('logs', args.dataset, args.saved_time, 'saved_model'), args)
    model = load_model(ckpt, config)
    ld_kwargs = SimpleNamespace(model_path=args.model_path, dataset=args.dataset)
    logger.info('Evaluate model on the generation task.')
    start_time = time.time()
    gen_out_name = 'eval_gen.pt'

    if not os.path.exists(ld_kwargs.model_path+'/'+gen_out_name):
        logger.info("Generating materials ...")
        (frac_coords, num_atoms, atom_types, matrices) = generation(model, ld_kwargs, args.num_samples, conds)
        torch.save({
            'eval_setting': args,
            'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'matrices': matrices,
            'time': time.time() - start_time
        }, model_path / gen_out_name)
        logger.info("Generation over.")
    else:
        logger.info("Have generated materials already!")
