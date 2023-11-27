import numpy as np
from estimate_homography import convert_to_homogenous_crd

class OptimizeResult():
    
    def __init__(self, x=0, nint=0, success=True, message='', min_cost=0):
        
        self.x = x
        self.nint = nint
        self.success = success
        self.message = message
        self.min_cost = min_cost

    def __repr__(self):
        out = "######## \n solution x: {}\n No of iterations : {} \n Success: {} \n Message : {} \n Min Cost: {} \n ########".\
            format(self.x, self.nint, self.success, self.message, self.min_cost)

        return out


class OptimizeFunction:
    
    def __init__(self, fun, x0, jac, args=()):
       
        self.result = OptimizeResult(x=x0, nint=0, success=True, message="Initialization", min_cost=0)
        self.x0 = x0
        self.args = args
        self.fun = fun
        self.jac = jac

    def levenberg_marquardt(self, delta_thresh=10**-16, tau=0.5):

        init_jac_f = self.jac(self.x0, *self.args)

        mu_k = tau * np.amax(np.diag(init_jac_f))

        xk = self.x0

        iter = 0

        update_iter = 0

        residual_k = self.fun(xk, *self.args)
        cost_k = np.dot(residual_k.T, residual_k)

        self.result.update_iter = update_iter
        self.result.min_cost = cost_k

        while True:

            jac_f = self.jac(xk, *self.args)

            delta_k = np.dot(jac_f.T, jac_f) + mu_k * np.eye(jac_f.shape[1], jac_f.shape[1])  
            delta_k = np.linalg.inv(delta_k)  
            delta_k = np.dot(delta_k, -1*jac_f.T)
            delta_k = np.dot(delta_k, residual_k)

            
            if np.linalg.norm(delta_k) < delta_thresh or (update_iter > 100):
            
                self.result.x = xk
                self.result.nint = iter
                self.result.update_iter = update_iter
                self.result.message = '||Delta_k|| < {}'.format(delta_thresh)
                self.result.success = True
                self.result.min_cost = cost_k
                return self.result

            xk_1 = xk + delta_k

            residual_k_1 = self.fun(xk_1, *self.args)

            cost_k_1 = np.dot(residual_k_1.T, residual_k_1)

            num = (cost_k - cost_k_1)
            den = np.dot(np.dot(delta_k.T, -1*jac_f.T), residual_k)
            den = den + np.dot(np.dot(delta_k.T, mu_k * np.eye(jac_f.shape[1], jac_f.shape[1])), delta_k)
            rho_LM = num/den

            mu_k = mu_k * max(1/3, 1 - (2 * rho_LM - 1)**3)

            if cost_k_1 < cost_k:
                
                xk = xk_1
                update_iter += 1
                residual_k = residual_k_1
                cost_k = cost_k_1


            iter += 1


    def dogleg(self):
        pass

    def gauss_newton(self):
        pass

    def gradient_descent(self):
        pass

def fun_LM_homography(h, x, x_dash):
   
    H = np.reshape(h, (3,3))

    x_tild = convert_to_homogenous_crd(x, axis=1)  
    x_tild = np.dot(H, x_tild.T)
    x_tild = x_tild/x_tild[-1, :]
    x_tild = x_tild.T  
    x_tild = x_tild[:, 0:2]

    residual = x_dash.flatten() - x_tild.flatten()  
    return residual


def jac_LM_homography(h, x, x_dash):

    def jac_fun1(inp_x, inp_h):
        
        num = inp_h[0] * inp_x[0] + inp_h[1] * inp_x[1] + inp_h[2]
        den = inp_h[6] * inp_x[0] + inp_h[7] * inp_x[1] + inp_h[8]

        out = np.zeros_like(inp_h)
        out[0] = -1 * inp_x[0]/den  
        out[1] = -1 * inp_x[1]/den  
        out[2] = -1/den  
        out[6] = (num * inp_x[0])/(den**2)  
        out[7] = (num * inp_x[1])/(den**2)  
        out[8] = num/(den**2)  

        return out

    def jac_fun2(inp_x, inp_h):
        
        num = inp_h[3] * inp_x[0] + inp_h[4] * inp_x[1] + inp_h[5]
        den = inp_h[6] * inp_x[0] + inp_h[7] * inp_x[1] + inp_h[8]

        out = np.zeros_like(inp_h)
        out[3] = -1 * inp_x[0] / den  
        out[4] = -1 * inp_x[1] / den  
        out[5] = -1 / den  
        out[6] = (num * inp_x[0])/(den ** 2)  
        out[7] = (num * inp_x[1])/(den ** 2)  
        out[8] = num/(den ** 2)  

        return out

    jac_eps_1 = np.apply_along_axis(jac_fun1, 1, x, h)
    jac_eps_2 = np.apply_along_axis(jac_fun2, 1, x, h)

    jac_out = np.empty((jac_eps_1.shape[0] + jac_eps_2.shape[0], jac_eps_1.shape[1]))
    jac_out[0::2] = jac_eps_1
    jac_out[1::2] = jac_eps_2

    return jac_out


def func(x):
    return np.array([x[0] + 0.5 * (x[0] - x[1])**3 - 1.0,
            0.5 * (x[1] - x[0])**3 + x[1]])

def jac(x):
    return np.array([[1 + 1.5 * (x[0] - x[1])**2,
                      -1.5 * (x[0] - x[1])**2],
                     [-1.5 * (x[1] - x[0])**2,
                      1 + 1.5 * (x[1] - x[0])**2]])

# if __name__ == "__main__":
#     from scipy import optimize
#     sol = optimize.least_squares(func, [0, 0], jac=jac, method='lm')
#     print(sol)
#     print("-----")


#     opt_obj = OptimizeFunction(fun=func, x0=np.array([0,0]), jac=jac)
#     LM_sol = opt_obj.levenberg_marquardt(delta_thresh=1e-6, tau=0.8)
#     print(LM_sol)

    
#     x_img1 = np.random.randint(20, 50, size=(20, 2))
#     H = np.arange(1, 10).reshape(3,3)

#     x_temp = convert_to_homogenous_crd(x_img1, axis=1)
#     x_tild = np.dot(H, x_temp.T)
#     x_tild = x_tild / x_tild[-1, :]
#     x_tild = x_tild.T
#     x_img2 = x_tild[:, 0:2]

#     x_inp = np.concatenate((x_img1, x_img2), axis=1)

#     H_noise = H + np.random.randn(3, 3) + 10

#     opt_obj = OptimizeFunction(fun=fun_LM_homography, x0=H_noise.flatten(), jac=jac_LM_homography,
#                                args=(x_inp[:, 0:2], x_inp[:, 2:]))
#     LM_sol = opt_obj.levenberg_marquardt(delta_thresh=1e-24, tau=0.8)

#     print("LM_sol : {}, \n &&&& {} &&&&".format(LM_sol, LM_sol.x/LM_sol.x[-1]))

#     print("=========")

#     sol = optimize.least_squares(fun_LM_homography, H_noise.flatten(), args=(x_inp[:, 0:2], x_inp[:, 2:]), method='lm',
#                                  jac=jac_LM_homography,
#                                  xtol=1e-24, ftol=1e-24)
#     print("scipy sol: \n &&& {}  &&&&".format(sol, sol.x/sol.x[-1]))

#     print("=========")

#     print("Original : {}".format(H/H[-1, -1]))
