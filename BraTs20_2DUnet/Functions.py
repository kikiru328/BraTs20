def Data_Preprocessing(modalities_dir):
    import nibabel as nib
    import numpy as np
    
    all_modalities = []    
    for modality in modalities_dir:      
        nifti_file   = nib.load(modality)
        brain_numpy  = np.asarray(nifti_file.dataobj)    
        all_modalities.append(brain_numpy)
    brain_affine   = nifti_file.affine
    all_modalities = np.array(all_modalities)
    all_modalities = np.rint(all_modalities).astype(np.int16)
    all_modalities = all_modalities[:, :, :, :]
    all_modalities = np.transpose(all_modalities)
    return all_modalities

def Data_Concatenate(Input_Data):
    import numpy as np
    counter=0
    Output= []
    for i in range(5):
        c=0
        counter=0
        for ii in range(len(Input_Data)):
            if (counter != len(Input_Data)):
                a= Input_Data[counter][:,:,:,i]
                b= Input_Data[counter+1][:,:,:,i]
                if(counter==0):
                    c= np.concatenate((a, b), axis=0)
                    counter= counter+2
                else:
                    c1= np.concatenate((a, b), axis=0)
                    c= np.concatenate((c, c1), axis=0)
                    counter= counter+2
        c= c[:,:,:,np.newaxis]
        Output.append(c)
    return Output