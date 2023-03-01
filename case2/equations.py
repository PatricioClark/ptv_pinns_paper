import tensorflow as tf
tf.keras.backend.set_floatx('float32')
import numpy as np

@tf.function
def NS3D(model, coords, params, separate_terms=False):
    """ NS 3D equations """

    PX    = params[0]
    nu    = params[1]

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(coords)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(coords)
            Yp = model(coords)[0]
            u  = Yp[:,0] 
            v  = Yp[:,1] 
            w  = Yp[:,2] 
            p  = Yp[:,3]

        # First derivatives
        grad_u = tape1.gradient(u, coords)
        u_t = grad_u[:,0]
        u_x = grad_u[:,1]
        u_y = grad_u[:,2]
        u_z = grad_u[:,3]

        grad_v = tape1.gradient(v, coords)
        v_t = grad_v[:,0]
        v_x = grad_v[:,1]
        v_y = grad_v[:,2]
        v_z = grad_v[:,3]

        grad_w = tape1.gradient(w, coords)
        w_t = grad_w[:,0]
        w_x = grad_w[:,1]
        w_y = grad_w[:,2]
        w_z = grad_w[:,3]

        grad_p = tape1.gradient(p, coords)
        p_x = grad_p[:,1]
        p_y = grad_p[:,2]
        p_z = grad_p[:,3]
        del tape1

    # Second derivatives
    u_xx = tape2.gradient(u_x, coords)[:,1]
    v_xx = tape2.gradient(v_x, coords)[:,1]
    w_xx = tape2.gradient(w_x, coords)[:,1]

    u_yy = tape2.gradient(u_y, coords)[:,2]
    v_yy = tape2.gradient(v_y, coords)[:,2]
    w_yy = tape2.gradient(w_y, coords)[:,2]

    u_zz = tape2.gradient(u_z, coords)[:,3]
    v_zz = tape2.gradient(v_z, coords)[:,3]
    w_zz = tape2.gradient(w_z, coords)[:,3]
    del tape2

    # Equations to be enforced
    if not separate_terms:
        f0 = u_x + v_y + w_z
        f1 = (u_t + u*u_x + v*u_y + w*u_z + p_x + PX - nu*(u_xx+u_yy+u_zz))
        f2 = (v_t + u*v_x + v*v_y + w*v_z + p_y      - nu*(v_xx+v_yy+v_zz))
        f3 = (w_t + u*w_x + v*w_y + w*w_z + p_z      - nu*(w_xx+w_yy+w_zz))
            
        return [f0, f1, f2, f3]
    else:
        return ([u_x, v_y, w_z],
                [u_t,
                 u*u_x, v*u_y, w*u_z,
                 p_x, PX*tf.ones(p_x.shape, dtype=p_x.dtype),
                -nu*u_xx, -nu*u_yy, -nu*u_zz],
                [v_t,
                 u*v_x, v*v_y, w*v_z,
                 p_y, 0*tf.ones(p_y.shape, dtype=p_y.dtype),
                -nu*v_xx, -nu*v_yy, -nu*v_zz],
                [w_t,
                 u*w_x, v*w_y, w*w_z,
                 p_z, 0*tf.ones(p_z.shape, dtype=p_z.dtype),
                -nu*w_xx, -nu*w_yy, -nu*w_zz],
                )
