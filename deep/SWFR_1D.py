"""
SWFR_1D.py
The code to solve spherical Wasserstein-Fisher-Rao metric on a real line.

Code is a copy of Yang Jing's code adaped from:
https://github.com/sharkjingyang/WFR

ToDo:
Figure out what aaaa1, costC2 are in the compute_loss_1 function

"""

import numpy as np
import math
import torch
import os
from torch.nn.functional import pad

device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")

##     ## ######## ##       ########  ######## ########
##     ## ##       ##       ##     ## ##       ##     ##
##     ## ##       ##       ##     ## ##       ##     ##
######### ######   ##       ########  ######   ########
##     ## ##       ##       ##        ##       ##   ##
##     ## ##       ##       ##        ##       ##    ##
##     ## ######## ######## ##        ######## ##     ##


def cvt(x):
    return x.to(device, non_blocking=True)  # convert tensor to device


def getv0(x, net, t):

    nex, d_extra = x.shape
    d = d_extra - 6
    z = pad(x[:, :d], (0, 1, 0, 0), value=t)  # concatenate with the time t

    gradPhi = net.trHess(z, justGrad=True)

    return -gradPhi[:, 0:d]


#######  ########  ########     ######   #######  ##       ##     ## ######## ########
##     ## ##     ## ##          ##    ## ##     ## ##       ##     ## ##       ##     ##
##     ## ##     ## ##          ##       ##     ## ##       ##     ## ##       ##     ##
##     ## ##     ## ######       ######  ##     ## ##       ##     ## ######   ########
##     ## ##     ## ##                ## ##     ## ##        ##   ##  ##       ##   ##
##     ## ##     ## ##          ##    ## ##     ## ##         ## ##   ##       ##    ##
#######  ########  ########     ######   #######  ########    ###    ######## ##     ##


def stepRK4(odefun, z, Phi, t0, t1, v0, alphaa=1):
    """
        Runge-Kutta 4 integration scheme

        Solves the ODE dz/dt = odefun(z, t, Phi, v0, alphaa) from time t0 to t1 with initial condition z0.

    :param odefun: function to apply at every time step
    :param z:      tensor nex-by-d+3, inputs
    :param Phi:    Module, the Phi potential function
    :param alph:   list, the 3 alpha values for the OT-Flow Problem
    :param t0:     float, starting time
    :param t1:     float, end time
    :return: tensor nex-by-d+3, features at time t1
    """

    h = t1 - t0  # step size
    z0 = z

    K = h * odefun(z0, t0, Phi, v0, alphaa=alphaa)
    z = z0 + (1.0 / 6.0) * K

    K = h * odefun(z0 + 0.5 * K, t0 + (h / 2), Phi, v0, alphaa=alphaa)
    z += (2.0 / 6.0) * K

    K = h * odefun(z0 + 0.5 * K, t0 + (h / 2), Phi, v0, alphaa=alphaa)
    z += (2.0 / 6.0) * K

    K = h * odefun(z0 + K, t0 + h, Phi, v0, alphaa=alphaa)
    z += (1.0 / 6.0) * K

    return z


def odefun(x, t, net, v0, alphaa=1):
    """
    The function chracterizing the neural ODE (Equation 4.9) along with accumulation of quantities

    The ODE is given by:

    d_t  [x ; l ; p ; s ; v ; s2 ; w] = odefun( [x ; l ; p ; s ; v ; s2 ; w] , t )

    Here,
    x - particle position
    l - log determinant
    p - accumulated regularization J_R (Equation 4.8)
    s - accumulated (-1/alpha) * phi-bar (Integral of 1/alpha * phi-bar(t)dt)
    v - accumulated SWFR energy J_WFR (The second term in Equation 4.8)
    s2 - accumulated (-1/alpha) * phi
    w - the weights for each sample

    Args:
        x (tensor of shape (nex, d+6)): input tensor with nex samples. The first d columns are the particle positions,\
        and the next 6 columns are the quantities described in the function description in that order.
        t: float, time
        net: Module, the neural network
        v0: tensor, nex-by-d, the initial velocity
        alphaa: float, the alpha value for the OT-Flow Problem

    """
    nex, d_extra = x.shape
    d = d_extra - 6

    z = pad(
        x[:, :d], (0, 1, 0, 0), value=t
    )  # concatenate with the time t. It adds a column to the right of z, filled with t
    X_data = x

    w_data = torch.ones(X_data.shape[0]).to(device).squeeze().float()  # 初始weight

    # exp(integral of (-1/alpha) * (phi(x,t) - phi-bar(t)dt) * weight
    unnorm = torch.exp(x[:, -4] + x[:, -2]) * w_data
    # w(x,t) = w(x,0) e^(-1/alpha * (phi(x,t) - phi-bar(t)), Page 12 of the paper
    # scaling the weights so that it sums to the number of samples, Equation (4.10)
    new_weight = unnorm / torch.sum(unnorm) * x.shape[0]

    Vx0 = net(z)

    ds0 = torch.dot(Vx0.squeeze(), new_weight / x.shape[0])
    Vx = Vx0 - ds0

    gradPhi, trH = net.trHess(z)

    dx = -gradPhi[:, 0:d]
    dl = -trH.unsqueeze(1)
    dp = torch.mul(
        new_weight.unsqueeze(-1) / x.shape[0],
        torch.sum(
            torch.pow(dx * new_weight.unsqueeze(-1) - v0 * w_data.reshape(-1, 1), 2),
            1,
            keepdims=True,
        ),
    )

    ds = (1 / alphaa * ds0) * torch.ones_like(dl)
    dv = torch.mul(
        0.5 * (torch.sum(torch.pow(dx, 2), 1, keepdims=True) + 1 / alphaa * (Vx**2)),
        new_weight.unsqueeze(-1) / x.shape[0],
    )
    ds2 = -1 / alphaa * Vx0
    dw = -1 / alphaa * torch.mul(Vx, new_weight.unsqueeze(-1))  # unormalized

    return torch.cat((dx, dl, dp, ds, dv, ds2, dw), 1)


def stepRK4_new(odefun_new, z, Phi, t0, t1, alphaa):

    h = t1 - t0  # step size
    z0 = z

    K = h * odefun_new(z0, t0, Phi, alphaa=alphaa)
    z = z0 + (1.0 / 6.0) * K

    K = h * odefun_new(z0 + 0.5 * K, t0 + (h / 2), Phi, alphaa=alphaa)
    z += (2.0 / 6.0) * K

    K = h * odefun_new(z0 + 0.5 * K, t0 + (h / 2), Phi, alphaa=alphaa)
    z += (2.0 / 6.0) * K

    K = h * odefun_new(z0 + K, t0 + h, Phi, alphaa=alphaa)
    z += (1.0 / 6.0) * K

    return z


def odefun_new(x, t, net, alphaa):

    d = 1
    z = pad(x[:, :d], (0, 1, 0, 0), value=t)  # concatenate with the time t
    Vx0 = net(z)
    gradPhi, trH = net.trHess(z)

    dx = -gradPhi[:, 0:d]
    ds2 = -1 / alphaa * Vx0

    return torch.cat((dx, ds2), 1)


######   #######   ######  ########    ######## ##     ## ##    ##  ######
##    ## ##     ## ##    ##    ##       ##       ##     ## ###   ## ##    ##
##       ##     ## ##          ##       ##       ##     ## ####  ## ##
##       ##     ##  ######     ##       ######   ##     ## ## ## ## ##
##       ##     ##       ##    ##       ##       ##     ## ##  #### ##
##    ## ##     ## ##    ##    ##       ##       ##     ## ##   ### ##    ##
######   #######   ######     ##       ##        #######  ##    ##  ######


def OTFlowProblem_1d(
    x, Phi, tspan, nt, alph=[1.0, 1.0, 1.0], alphaa=1, jjj=0, device=device
):

    w_data = torch.ones(x.shape[0]).to(device).squeeze().float()

    h = (tspan[1] - tspan[0]) / nt

    z = pad(x, (0, 5), value=0)

    z = torch.cat((z, w_data.reshape(-1, 1)), dim=1)

    tk = tspan[0]

    v0 = getv0(z, Phi, tspan[0])
    for k in range(nt):
        z = stepRK4(odefun, z, Phi, tk, tk + h, v0, alphaa=alphaa)
        tk += h

    new_weight = torch.exp(z[:, -4] + z[:, -2]) * w_data
    new_weight = new_weight / torch.sum(new_weight) * new_weight.shape[0]

    # Total SWFR energy over all samples
    costL = torch.sum(z[:, -3])  # dv, 0.5*rho*v^2+0.5*alpha*rho*g^2

    d = z.shape[1] - 6
    l = z[:, d]  # log-det

    # Calculation of the terminal condition KL term (Equation 4.4)
    # We have three different cases for this term:
    # jjj = 0 : the standard normal distribution as terminal cond. with the term for phi_abr omitted.
    # jjj = 1: Only the accumulation of the phi-bar term.
    # jjj = 2: The full KL term with phi_bar.
    if jjj == 0:
        costC = (
            0.5 * d * math.log(2 * math.pi)
            - torch.dot(l, w_data / w_data.shape[0])
            + 0.5
            * torch.dot(
                torch.sum(torch.pow(z[:, 0:d], 2), 1, keepdims=True).squeeze(),
                w_data / w_data.shape[0],
            )
            + torch.dot(z[:, -2], w_data / w_data.shape[0])
        )
    elif jjj == 1:
        costC = -torch.dot(z[:, -4], w_data / w_data.shape[0])
    elif jjj == 2:
        costC = (
            0.5 * d * math.log(2 * math.pi)
            - torch.mean(l)
            + 0.5 * torch.mean(torch.sum(torch.pow(z[:, 0:d], 2), 1, keepdims=True))
            + torch.mean(z[:, -2])
            + z[0, -4]
        )

    costV = torch.mean(z[:, -5])  # total sum of the regularization term
    cs = [costL, costC, costV]

    # costC: KL term
    # costL: SWFR energy
    # costV: Regularization term

    return costC, costL, costV, cs, new_weight, z[0, -4]


def compute_loss_1(net, x, x2, nt):
    alphaa = 1
    costC1, costL1, costV1, cs, weights, aaaa1 = OTFlowProblem_1d(
        x,
        net,
        [0, 1],
        nt=nt,
        stepper="rk4",
        alph=[1.0, 1.0, 1.0],
        alphaa=alphaa,
        jjj=0,
        device=device,
    )

    tspan = [1, 0]
    h = (tspan[1] - tspan[0]) / nt

    z2 = pad(x2.unsqueeze(-1), (0, 1, 0, 0), value=0)

    tk = tspan[0]

    # To find the remaining term for the KL term, solve ODE (4.11)
    for k in range(nt):
        z2 = stepRK4_new(odefun_new, z2, net, tk, tk + h, alphaa=alphaa)
        tk += h

    # Log of mean of accumulation of exp(-(1/a)*phi) over all samples
    # I am not sure if this showed up in the paper
    costC2 = torch.log(torch.mean(torch.exp(z2[:, -1])))

    Jc = (
        (costC1 + costC2) * 100
        + costL1 * 10
        + costV1 * 1
        + torch.abs(aaaa1 - costC2) * 1
    )
    cs[1] = cs[1] + costC2
    return Jc, cs, weights, 0


######   ########  #######  ########  ########  ######  ####  ######
##    ##  ##       ##     ## ##     ## ##       ##    ##  ##  ##    ##
##        ##       ##     ## ##     ## ##       ##        ##  ##
##   #### ######   ##     ## ##     ## ######    ######   ##  ##
##    ##  ##       ##     ## ##     ## ##             ##  ##  ##
##    ##  ##       ##     ## ##     ## ##       ##    ##  ##  ##    ##
######   ########  #######  ########  ########  ######  ####  ######

if __name__ == "__main__":

    # Precision
    prec = torch.float32

    torch.set_default_dtype(prec)

    # neural network for the potential function Phi
    d = 1
    alph = 10
    nt = 8  # number of time steps
    nt_val = 8  # number of time steps for validation
    nTh = 2
    m = 32
    net = Phi(nTh=nTh, m=args.m, d=d, alph=alph)
    net = net.to(prec).to(device)

    optim = torch.optim.Adam(
        net.parameters(), lr=0.05, weight_decay=0.0
    )  # lr=0.04 good

    logger.info(net)
    logger.info("-------------------------")
    logger.info("DIMENSION={:}  m={:}  nTh={:}   alpha={:}".format(d, m, nTh, alph))
    logger.info("nt={:}   nt_val={:}".format(nt, nt_val))
    logger.info("Number of trainable parameters: {}".format(count_parameters(net)))
    logger.info("-------------------------")
    logger.info(str(optim))  # optimizer info
    logger.info(
        "data={:} batch_size={:} gpu={:}".format(args.data, args.batch_size, args.gpu)
    )
    logger.info(
        "maxIters={:} val_freq={:} viz_freq={:}".format(
            args.niters, args.val_freq, args.viz_freq
        )
    )
    logger.info("saveLocation = {:}".format(args.save))
    logger.info("-------------------------\n")

    end = time.time()
    best_loss = float("inf")
    bestParams = None

    # setup data [nSamples, d]
    # use one batch as the entire data set

    print(args.data)
    X_data, _ = toy_data.inf_train_gen(
        args.data, batch_size=args.batch_size, require_density=False
    )
    x0 = torch.from_numpy(X_data).to(device).squeeze().float()

    x2 = torch.randn(args.batch_size).to(device)
    log_msg = "{:5s}  {:6s}   {:9s}  {:9s}  {:9s}  {:9s} {:9s}     {:9s}  {:9s}  {:9s}  {:9s} {:9s} ".format(
        "iter",
        " time",
        "loss",
        "L (L_2)",
        "C (loss)",
        "R (HJB)",
        "V(velocity)",
        "valLoss",
        "valL",
        "valC",
        "valR",
        "valV",
    )

    logger.info(log_msg)

    time_meter = utils.AverageMeter()

    net.train()
    MMD_list = []
    ITR = []
    Time = []
    MSE = []
    DEVI = []
    t = 0

    costL_list = []
    for itr in range(1, args.niters + 1):
        # train
        optim.zero_grad()

        loss, costs, weights, devi = compute_loss_1(net, x0, x2, nt=args.nt)
        loss.backward()
        optim.step()
        time_meter.update(time.time() - end)

        log_message = "{:05d}  {:6.3f}   {:9.5f}  {:9.5f}  {:9.5f}  {:9.5f} ".format(
            itr, time_meter.val, loss, costs[0], costs[1], costs[2]
        )
        t = t + time_meter.val
        costL_list.append(float(costs[0]))

        # with torch.no_grad():
        #     net.eval()
        #
        #     genModel, _ = integrate_ex(y1[:, 0:d], net, [1, 0.0], nt_val, stepper="rk4", alph=net.alph,
        #                                alphaa=args.alphaa)
        #     mmd1 = MMD_Weighted(x0.unsqueeze(-1), genModel[:, 0].unsqueeze(-1), genModel[:, -1].unsqueeze(-1))
        #
        #     total_weights = torch.sum(genModel[:, -1])
        #
        #
        #     MMD_list.append(float(mmd1))
        #     ITR.append(itr)
        #     Time.append(t)

        # MSE.append(float(mse))
        # DEVI.append(devi)

        # save best set of parameters
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_costs = costs
            utils.makedirs(args.save)
            best_params = net.state_dict()
            torch.save(
                {
                    "args": args,
                    "state_dict": best_params,
                },
                os.path.join(
                    args.save,
                    start_time
                    + "_{:}_alph{:}_{:}_m{:}_checkpt.pth".format(
                        args.data, int(alph[1]), int(alph[2]), m
                    ),
                ),
            )
            net.train()

        if itr % 100 == 0:
            with torch.no_grad():
                net.eval()
                x0val, rho_x0val = toy_data.inf_train_gen(
                    args.data, batch_size=10000, require_density=False
                )
                x0val = cvt(torch.from_numpy(x0val))
                x3 = torch.randn(args.batch_size).to(device)
                test_loss, test_costs, test_weights, test_devi = compute_loss_1(
                    net, x0val, x3, nt=30
                )
                log_message = "    1/2rho v^2+1/2 alpha rho g^2 {:9.5f}  ".format(
                    test_costs[0]
                )
                net.train()

        # validate
        if itr % args.val_freq == 0:
            with torch.no_grad():
                net.eval()

                x0val, rho_x0val = toy_data.inf_train_gen(
                    args.data, batch_size=args.batch_size, require_density=False
                )
                x0val = cvt(torch.from_numpy(x0val))
                x3 = torch.randn(args.batch_size).to(device)

                test_loss, test_costs, test_weights, test_devi = compute_loss_1(
                    net, x0val, x3, nt=nt_val
                )

                nSamples = args.batch_size
                y1 = cvt(torch.randn(nSamples, d))
                genModel, _ = integrate_ex(
                    y1[:, 0:d],
                    net,
                    [1, 0.0],
                    nt_val,
                    stepper="rk4",
                    alph=net.alph,
                    alphaa=args.alphaa,
                )

                # mmd1 = MMD_Weighted(x0val.unsqueeze(-1), genModel[:, 0].unsqueeze(-1), genModel[:, -1].unsqueeze(-1))

                total_weights = torch.sum(genModel[:, -1])
                # print('total_weights:', total_weights)
                # mse = (torch.dot(genModel[:,0]**2, genModel[:,-1])/total_weights-10)**2

                # add to print message
                log_message += "    {:9.3e}  {:9.3e}  {:9.3e}  {:9.3e} {:9.3e} ".format(
                    test_loss, test_costs[0], test_costs[1], test_costs[2], 0
                )

                ITR.append(itr)
                Time.append(t)

                # save best set of parameters
                if test_loss.item() < best_loss:
                    best_loss = test_loss.item()
                    best_costs = test_costs
                    utils.makedirs(args.save)
                    best_params = net.state_dict()
                    torch.save(
                        {
                            "args": args,
                            "state_dict": best_params,
                        },
                        os.path.join(
                            args.save,
                            start_time
                            + "_{:}_alph{:}_{:}_m{:}_checkpt.pth".format(
                                args.data, int(alph[1]), int(alph[2]), m
                            ),
                        ),
                    )
                    net.train()

        logger.info(log_message)

        # create plots
        if itr % args.viz_freq == 0:
            with torch.no_grad():
                net.eval()
                curr_state = net.state_dict()
                net.load_state_dict(best_params)

                nSamples = 100000
                p_samples, _ = cvt(
                    torch.Tensor(
                        toy_data.inf_train_gen(
                            args.data, batch_size=nSamples, require_density=False
                        )
                    )
                )
                y = cvt(
                    torch.randn(nSamples, d)
                )  # sampling from the standard normal (rho_1)

                sPath_1 = os.path.join(
                    args.save, "figs", start_time + "_{:04d}.png".format(itr)
                )
                sPath_2 = os.path.join(
                    args.save, "figs", start_time + "_{:04d}_time.png".format(itr)
                )
                sPath_3 = os.path.join(
                    args.save, "figs", start_time + "_{:04d}_weight_1.png".format(itr)
                )
                sPath_4 = os.path.join(
                    args.save, "figs", start_time + "_{:04d}_weight_2.png".format(itr)
                )
                sPath_5 = os.path.join(
                    args.save, "figs", start_time + "_{:04d}_unweighted.png".format(itr)
                )

                plot1d(
                    net,
                    p_samples,
                    y,
                    nt_val,
                    sPath_1,
                    sPath_2,
                    sPath_3,
                    sPath_4,
                    sPath_5,
                    doPaths=True,
                    sTitle="{:s}  -  loss {:.2f}  ,  C {:.2f}  ,  alph {:.1f} {:.1f}  "
                    " nt {:d}   m {:d}  nTh {:d}  ".format(
                        args.data,
                        best_loss,
                        best_costs[1],
                        alph[1],
                        alph[2],
                        nt,
                        m,
                        nTh,
                    ),
                    alphaa=args.alphaa,
                )

                net.load_state_dict(curr_state)
                net.train()

        # shrink step size
        if itr % args.drop_freq == 0:
            for p in optim.param_groups:
                p["lr"] /= args.lr_drop
            print("lr: ", p["lr"])

        # resample data
        if itr % args.sample_freq == 0:
            logger.info("resampling")
            x0, rho_x0 = toy_data.inf_train_gen(
                args.data, batch_size=args.batch_size, require_density=False
            )
            x0 = cvt(torch.from_numpy(x0))
            x2 = torch.randn(args.batch_size).to(device)

        end = time.time()

    # print('costL_list:', costL_list)
    # print(ITR)

    # print('Inverse_Flow_MMD',MMD_list)
    # print('Inverse_Flow_MSE',MSE)
    print(Time)
    logger.info("Training Time: {:} seconds".format(time_meter.sum))
    logger.info(
        "Training has finished.  "
        + start_time
        + "_{:}_alph{:}_{:}_m{:}_checkpt.pth".format(
            args.data, int(alph[1]), int(alph[2]), m
        )
    )
