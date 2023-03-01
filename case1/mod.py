# Functions used by run_pinn.py

import numpy as np
import h5py

def generate_data(params):
    try:
        X_data = np.load(params.paths.odir+'X_data.npy')
        Y_data = np.load(params.paths.odir+'Y_data.npy')
        return X_data, Y_data
    except:
        Nt   = params.Nt
        dt   = params.dt

        # Read file
        data = np.loadtxt('data/velos.dat')

        # Shuffle
        np.random.shuffle(data)

        # Unpack and reshape
        t_d  = data[:,0:1]
        x_d  = data[:,1:2]
        y_d  = data[:,2:3]
        z_d  = data[:,3:4]
        u_d  = data[:,4:5]
        v_d  = data[:,5:6]
        w_d  = data[:,6:7]
        p_d  = 0.0*w_d

        X_data = np.concatenate((t_d, x_d, y_d, z_d), 1)
        Y_data = np.concatenate((u_d, v_d, w_d, p_d), 1)

        X_data = X_data.astype(np.float32)
        Y_data = Y_data.astype(np.float32)

        np.save(params.paths.odir+'X_data.npy', X_data)
        np.save(params.paths.odir+'Y_data.npy', Y_data)

        return X_data, Y_data

def plot_points(params):
    Xg = np.arange(0, params.delta*params.Nx, params.delta)
    Yg = np.arange(0, params.delta*params.Ny, params.delta)
    Zg = np.arange(0, params.delta*params.Nz, params.delta)

    T = 4.0*params.dt
    X = Xg*params.vl + params.x0
    Y = Yg*params.vl + params.y0
    Z = Zg*params.vl + params.z0

    T, X, Y, Z = np.meshgrid(T, X, Y, Z, indexing='ij')

    T = T.reshape(-1,1)
    X = X.reshape(-1,1)
    Y = Y.reshape(-1,1)
    Z = Z.reshape(-1,1)

    X = np.concatenate((T, X, Y, Z), 1)
    return X.astype(np.float32)

def dns_validation(self, params, Eqs, eq_params):
    def validation(ep):
        Nx = params.Nx
        Ny = params.Ny
        Nz = params.Nz

        # Points for evaluation
        X_plot = plot_points(params)

        # Get predicted
        Y   = self.model(X_plot)[0].numpy()
        u_p = Y[:,0].reshape((Nx,Ny,Nz))
        v_p = Y[:,1].reshape((Nx,Ny,Nz))
        w_p = Y[:,2].reshape((Nx,Ny,Nz))
        p_p = Y[:,3].reshape((Nx,Ny,Nz))
        p_p = p_p - np.mean(p_p)

        # Save predictions
        np.save('predicted.npy', np.array([u_p, v_p, w_p, p_p]))

        # Get data
        st  = str(params.startTime)
        if st.endswith('.0'):
            st = st[:-2]
        dns = h5py.File(f'../../../dns/DNS_V_A_P_t{st}_5WU.mat', 'r')
        udns = np.transpose(dns['udns'][:])
        vdns = np.transpose(dns['vdns'][:])
        wdns = np.transpose(dns['wdns'][:])
        pdns = np.transpose(dns['pdns'][:])
        pdns = pdns - np.mean(pdns)
        dns.close()
        
        # Test errors
        errs_u = np.mean((u_p - udns)**2)/np.std(udns)**2
        errs_p = np.mean((p_p - pdns)**2)/np.std(pdns)**2

        # Correlations
        aux    = np.mean((u_p-np.mean(u_p))*(udns-np.mean(udns)))
        norm1  = np.std(u_p)
        norm2  = np.std(udns)
        corr_u = aux/(norm1*norm2)

        aux    = np.mean(p_p*pdns)
        norm1  = np.std(p_p)
        norm2  = np.std(pdns)
        corr_p = aux/(norm1*norm2)

        # RMS erros
        utau = 4.9968e-2
        rmse_u = np.sqrt(np.mean((u_p-udns)**2))/utau
        rmse_p = np.sqrt(np.mean((p_p-pdns)**2))/utau**2

        output_file = open(self.dest + 'validation.dat', 'a')
        print(ep,
              errs_u, rmse_u, corr_u,
              errs_p, rmse_p, corr_p,
              file=output_file)
        output_file.close()

    return validation

def plot_faces(ax,field,params,vmin=None,vmax=None):
    if vmin is None:
        vmin = field.min()
    if vmax is None:
        vmax = field.max()
    
    lvs = 30

    Xc = params.xs
    Yc = params.ys
    Zc = params.zs

    # this is the example that worked for you:
    # TOP
    X = Zc
    Y = Xc
    X, Y = np.meshgrid(X, Y)
    Z = field[:,-1,:]
    ax.contourf(X, Y, Z, zdir='z', offset=Yc[-1],
                          levels=np.linspace(vmin,vmax,lvs),cmap='jet')

    # now, for the x-constant face, assign the contour to the x-plot-variable:
    # pero que de verdad es z
    X = Yc
    Y = Xc
    X, Y = np.meshgrid(X, Y)
    Z = field[:,:,-1]
    ax.contourf(Z, Y, X, zdir='x', offset=Zc[-1],
                          levels=np.linspace(vmin,vmax,lvs),cmap='jet')

    # likewise, for the y-constant face, assign the contour to the y-plot-variable:
    X = Zc
    Y = Yc
    X, Y = np.meshgrid(X[:], Y)
    Z = field[0,:,:]
    cset = ax.contourf(X, Z, Y, zdir='y', offset=0,
                          levels=np.linspace(vmin,vmax,lvs),cmap='jet')

    # setting 3D-axis-limits:    
    ax.set_xlim3d(Zc[0], Zc[-1])
    ax.set_xlabel('$z^+$')
    ax.set_xlim3d(Yc[0], Yc[-1])
    ax.set_zlabel('$y^+$')
    ax.set_xlim3d(Xc[0], Xc[-1])
    ax.set_ylabel('$x^+$')
    ax.locator_params(axis='x', nbins=4)
    ax.locator_params(axis='y', nbins=4)
    ax.locator_params(axis='z', nbins=4)
    return cset
