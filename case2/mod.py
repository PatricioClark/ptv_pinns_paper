# Functions used by run_pinn.py

import numpy as np
import h5py

def inside_domain(x, y, z, params):
    xmax = np.max(params.xs) + params.dom_exp*params.dx
    xmin = np.min(params.xs) - params.dom_exp*params.dx
    ymax = np.max(params.ys) + params.dom_exp*params.dx
    ymin = np.min(params.ys)
    zmax = np.max(params.zs) + params.dom_exp*params.dx
    zmin = np.min(params.zs) - params.dom_exp*params.dx
    if (x<xmax and x>xmin and
        y<ymax and y>ymin and
        z<zmax and z>zmin):
        return True
    else:
        return False

def generate_data(params):
    try:
        X_data = np.load(params.paths.odir+'X_data.npy')
        Y_data = np.load(params.paths.odir+'Y_data.npy')
    except:
        Nt   = params.Nt
        dt   = params.dt
        vl   = params.vl
        t_ini = params.t_ini

        """Populate data arrays"""
        t_d, x_d, y_d, z_d = [], [], [], []
        u_d, v_d, w_d, p_d = [], [], [], []
        lambda_data, lambda_phys = [], []
        for tidx in range(t_ini, t_ini+Nt):
            data = np.loadtxt(f'data/velos.{tidx:04}.dat')

            # Shuffle
            np.random.shuffle(data)

            xc = data[:,0]
            yc = data[:,1]
            zc = data[:,2]
            uc = data[:,4]
            vc = data[:,5]
            wc = data[:,6]
            for ii in range(len(xc)):
                x = xc[ii]*vl
                y = yc[ii]*vl
                z = zc[ii]*vl

                if ((params.enforce_domain) and
                    (not inside_domain(x, y, z, params))):
                    continue

                t_d.append(tidx*dt)
                x_d.append(x)
                y_d.append(y)
                z_d.append(z)
                u_d.append(uc[ii])
                v_d.append(vc[ii])
                w_d.append(wc[ii])
                p_d.append(0.0)
                lambda_data.append(1.0)
                lambda_phys.append(1.0)

            idxs   = np.random.choice(params.Nx*params.Ny*params.Nz, len(xc))
            X_plot = plot_points(params, tidx)[idxs]
            for point in X_plot:
                t_d.append(point[0])
                x_d.append(point[1])
                y_d.append(point[2])
                z_d.append(point[3])
                u_d.append(0.0)
                v_d.append(0.0)
                w_d.append(0.0)
                p_d.append(0.0)
                lambda_data.append(0.0)
                lambda_phys.append(1.0)

        # Convert into arrays with correct shape
        t_d = np.array(t_d).reshape(-1,1)
        x_d = np.array(x_d).reshape(-1,1)
        y_d = np.array(y_d).reshape(-1,1)
        z_d = np.array(z_d).reshape(-1,1)
        u_d = np.array(u_d).reshape(-1,1)
        v_d = np.array(v_d).reshape(-1,1)
        w_d = np.array(w_d).reshape(-1,1)
        p_d = np.array(p_d).reshape(-1,1)

        X_data = np.concatenate((t_d, x_d, y_d, z_d), 1)
        Y_data = np.concatenate((u_d, v_d, w_d, p_d), 1)

        X_data = X_data.astype(np.float32)
        Y_data = Y_data.astype(np.float32)
        lambda_data = np.array(lambda_data).astype(np.float32)
        lambda_phys = np.array(lambda_phys).astype(np.float32)

        np.save('X_data.npy', X_data)
        np.save('Y_data.npy', Y_data)

        np.save('lambda_data.npy', lambda_data)
        np.save('lambda_phys.npy', lambda_phys)

    return X_data, Y_data, lambda_data, lambda_phys

def plot_points(params, tidx=0, k0=False, j0=False):
    Nx = params.Nx
    Ny = params.Ny
    Nz = params.Nz

    T = tidx*params.dt
    X = params.xs
    Y = params.ys
    Z = params.zs
    if k0:
        Z = Z[k0]
    if j0:
        Y = Y[j0]

    T, X, Y, Z = np.meshgrid(T, X, Y, Z, indexing='ij')

    T = T.reshape(-1,1)
    X = X.reshape(-1,1)
    Y = Y.reshape(-1,1)
    Z = Z.reshape(-1,1)

    X = np.concatenate((T, X, Y, Z), 1)
    return X.astype(np.float32)

def dns_validation(self, params):
    def validation(ep):
        Nt = params.Nt
        Nx = params.Nx
        Ny = params.Ny
        Nz = params.Nz
        dt = params.dt
        t0 = params.t0
        t_ini = params.t_ini

        # Get predicted
        X_plot = plot_points(params, tidx=int(Nt//2+t_ini))
        Y  = self.model(X_plot)[0].numpy()
        u_p = Y[:,0].reshape((Nx,Ny,Nz))
        v_p = Y[:,1].reshape((Nx,Ny,Nz))
        w_p = Y[:,2].reshape((Nx,Ny,Nz))
        p_p = Y[:,3].reshape((Nx,Ny,Nz))
        p_p = p_p - np.mean(p_p)

        # Save predicted
        np.save('predicted.npy', np.array([u_p, v_p, w_p, p_p]))
        
        # Get data
        st  = t0 + int(Nt//2+t_ini)*dt
        dns = h5py.File(f'dns/DNS_V_A_P_t{st:.5f}_5WU.mat', 'r')
        udns = np.transpose(dns['udns'][:])[:,::-1,:]
        vdns = np.transpose(dns['vdns'][:])[:,::-1,:]
        wdns = np.transpose(dns['wdns'][:])[:,::-1,:]
        pdns = np.transpose(dns['pdns'][:])[:,::-1,:]
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
