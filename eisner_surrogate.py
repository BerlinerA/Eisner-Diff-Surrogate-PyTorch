import torch
import numpy as np
from torch.autograd import Function, gradcheck
from scipy.special import softmax

from eisner import parse_proj


def eisner_surrogate(arc_scores, hard=False):
    ''' Parse using a differentiable surrogate of Eisner's algorithm.

    Parameters
    ----------
    arc_scores : Tensor
        Arc scores
    hard : bool
        True for discrete (valid) non-differentiable dependency trees,
        False for soft differentiable dependency trees (default is False).
    Returns
    -------
    Tensor
        The highest scoring hard/soft dependency tree
    '''
    return EisnerSurrogate.apply(arc_scores, hard)


class EisnerSurrogate(Function):

    @staticmethod
    def _inside(n, w, cw_arr, bp_arr, w_arr, hard):

        for l in range(1, n):
            for i in range(n - l):
                j = i + l

                # left to right incomplete
                w_right_i_i_j = \
                    cw_arr[0, i, i: j] \
                    + cw_arr[1, i + 1: j + 1, j]

                if hard:
                    bp_right_i_i_j = np.zeros_like(w_right_i_i_j)
                    bp_right_i_i_j[np.argmax(w_right_i_i_j)] = 1.
                else:
                    bp_right_i_i_j = softmax(w_right_i_i_j)

                right_i_term = np.dot(bp_right_i_i_j, w_right_i_i_j)
                w_arr[2, i, j, i: j] = w_right_i_i_j
                bp_arr[2, i, j, i: j] = bp_right_i_i_j
                cw_arr[2, i, j] = w[i, j] + right_i_term

                # right to left incomplete
                w_arr[3, i, j, i: j] = w_right_i_i_j
                bp_arr[3, i, j, i: j] = bp_right_i_i_j
                cw_arr[3, i, j] = w[j, i] + right_i_term

                # left to right complete
                w_right_c_i_j = \
                    cw_arr[2, i, i + 1: j + 1] \
                    + cw_arr[0, i + 1: j + 1, j]

                if hard:
                    bp_right_c_i_j = np.zeros_like(w_right_c_i_j)
                    bp_right_c_i_j[np.argmax(w_right_c_i_j)] = 1.
                else:
                    bp_right_c_i_j = softmax(w_right_c_i_j)

                w_arr[0, i, j, i + 1: j + 1] = w_right_c_i_j
                bp_arr[0, i, j, i + 1:j + 1] = bp_right_c_i_j
                cw_arr[0, i, j] = np.dot(bp_right_c_i_j, w_right_c_i_j)

                # right to left complete
                w_left_c_i_j = \
                    cw_arr[1, i, i: j] \
                    + cw_arr[3, i: j, j]

                if hard:
                    bp_left_c_i_j = np.zeros_like(w_left_c_i_j)
                    bp_left_c_i_j[np.argmax(w_left_c_i_j)] = 1.
                else:
                    bp_left_c_i_j = softmax(w_left_c_i_j)

                w_arr[1, i, j, i: j] = w_left_c_i_j
                bp_arr[1, i, j, i:j] = bp_left_c_i_j
                cw_arr[1, i, j] = np.dot(bp_left_c_i_j, w_left_c_i_j)

    @staticmethod
    def _backptr(n, soft_cw_arr, bp_arr):

        soft_cw_arr[0, 0, n - 1] = 1

        for l in range(n - 1, 0, -1):
            for i in range(0, n - l):
                j = i + l

                right_c_term = soft_cw_arr[0, i, j] * bp_arr[0, i, j, i + 1: j + 1]
                soft_cw_arr[2, i, i + 1: j + 1] += right_c_term
                soft_cw_arr[0, i + 1: j + 1, j] += right_c_term

                left_c_term = soft_cw_arr[1, i, j] * bp_arr[1, i, j, i: j]
                soft_cw_arr[1, i, i: j] += left_c_term
                soft_cw_arr[3, i: j, j] += left_c_term

                update_term = (soft_cw_arr[3, i, j] + soft_cw_arr[2, i, j]) \
                              * bp_arr[2, i, j, i: j]

                soft_cw_arr[0, i, i: j] += update_term
                soft_cw_arr[1, i + 1: j + 1, j] += update_term

    @staticmethod
    def forward(ctx, input, hard=False):

        w = input.cpu().detach().numpy()
        n = w.shape[-1]

        w_arr = np.zeros(shape=(4, n, n, n), dtype=np.float64)
        bp_arr = np.zeros_like(w_arr)
        cw_arr = np.zeros(shape=(4, n, n), dtype=np.float64)
        soft_cw_arr = np.zeros_like(cw_arr)

        # infer the highest scoring dependency tree
        EisnerSurrogate._inside(n, w, cw_arr, bp_arr, w_arr, hard)
        EisnerSurrogate._backptr(n, soft_cw_arr, bp_arr)

        d_tree = np.zeros_like(w)
        for i in range(n):
            for j in range(1, n):
                if i < j:
                    d_tree[i, j] = soft_cw_arr[2, i, j]
                elif j < i:
                    d_tree[i, j] = soft_cw_arr[3, j, i]

        # stash information for backward computation
        ctx.soft_cw = soft_cw_arr
        ctx.bp = bp_arr
        ctx.w = w_arr

        return input.new(d_tree)

    @staticmethod
    def backward(ctx, grad_output):

        # unpack stashed information
        soft_cw_arr, bp_arr, w_arr = ctx.soft_cw, ctx.bp, ctx.w

        np_grad_output = grad_output.cpu().detach().numpy()
        n = np_grad_output.shape[-1]

        g_w_arr = np.zeros(shape=(4, n, n, n), dtype=np.float64)
        g_bp_arr = np.zeros_like(g_w_arr)
        g_cw_arr = np.zeros(shape=(4, n, n), dtype=np.float64)
        g_soft_cw_arr = np.zeros_like(g_cw_arr)

        for i in range(n):
            for j in range(1, n):
                if i < j:
                    g_soft_cw_arr[2, i, j] = np_grad_output[i, j]
                elif j < i:
                    g_soft_cw_arr[3, j, i] = np_grad_output[i, j]

        # gradient computation
        EisnerSurrogate._backward_backptr(n, g_soft_cw_arr, g_bp_arr, soft_cw_arr, bp_arr)
        EisnerSurrogate._backward_inside(n, g_cw_arr, g_bp_arr, g_w_arr, bp_arr, w_arr)

        np_grad_input = np.zeros_like(np_grad_output)
        for i in range(n):
            for j in range(1, n):
                if i < j:
                    np_grad_input[i, j] = g_cw_arr[2, i, j]
                elif j < i:
                    np_grad_input[i, j] = g_cw_arr[3, j, i]

        return grad_output.new(np_grad_input), None

    @staticmethod
    def _backward_backptr(n, g_soft_cw_arr, g_bp_arr, soft_cw_arr, bp_arr):

        for l in range(1, n):
            for i in range(n - l):
                j = i + l

                # right to left incomplete
                update_term = np.dot(g_soft_cw_arr[0, i, i:j]
                                     + g_soft_cw_arr[1, i + 1: j + 1, j],
                                     bp_arr[3, i, j, i:j])
                g_soft_cw_arr[3, i, j] += update_term

                g_bp_arr[3, i, j, i:j] = \
                    (g_soft_cw_arr[0, i, i:j]
                     + g_soft_cw_arr[1, i + 1: j + 1, j]) \
                    * soft_cw_arr[3, i, j]

                # left to right incomplete
                g_soft_cw_arr[2, i, j] += update_term

                g_bp_arr[2, i, j, i:j] = \
                    (g_soft_cw_arr[0, i, i:j]
                     + g_soft_cw_arr[1, i + 1: j + 1, j]) \
                    * soft_cw_arr[2, i, j]

                # right to left complete
                g_soft_cw_arr[1, i, j] += \
                    np.dot(g_soft_cw_arr[1, i, i:j]
                           + g_soft_cw_arr[3, i:j, j],
                           bp_arr[1, i, j, i:j])

                g_bp_arr[1, i, j, i:j] = \
                    (g_soft_cw_arr[1, i, i:j]
                     + g_soft_cw_arr[3, i:j, j]) \
                    * soft_cw_arr[1, i, j]

                # left to right complete
                g_soft_cw_arr[0, i, j] += \
                    np.dot(g_soft_cw_arr[2, i, i + 1:j + 1]
                           + g_soft_cw_arr[0, i + 1: j + 1, j],
                           bp_arr[0, i, j, i + 1:j + 1])

                g_bp_arr[0, i, j, i + 1:j + 1] = \
                    (g_soft_cw_arr[2, i, i + 1:j + 1]
                     + g_soft_cw_arr[0, i + 1: j + 1, j]) \
                    * soft_cw_arr[0, i, j]

    @staticmethod
    def _backward_inside(n, g_cw_arr, g_bp_arr, g_w_arr, bp_arr, w_arr):

        for l in range(n - 1, 0, -1):
            for i in range(0, n - l):
                j = i + l

                # right to left complete
                g_bp_arr[1, i, j, i:j] += \
                    g_cw_arr[1, i, j] \
                    * w_arr[1, i, j, i:j]

                g_w_arr[1, i, j, i:j] += \
                    g_cw_arr[1, i, j] \
                    * bp_arr[1, i, j, i:j]

                s = np.dot(g_bp_arr[1, i, j, i:j],
                           bp_arr[1, i, j, i:j])

                g_w_arr[1, i, j, i:j] += \
                    bp_arr[1, i, j, i:j] \
                    * (g_bp_arr[1, i, j, i:j] - s)

                g_cw_arr[1, i, i:j] += g_w_arr[1, i, j, i:j]
                g_cw_arr[3, i:j, j] += g_w_arr[1, i, j, i:j]

                # left to right complete
                g_bp_arr[0, i, j, i + 1:j + 1] += \
                    g_cw_arr[0, i, j] \
                    * w_arr[0, i, j, i + 1:j + 1]

                g_w_arr[0, i, j, i + 1:j + 1] += \
                    g_cw_arr[0, i, j] \
                    * bp_arr[0, i, j, i + 1:j + 1]

                s = np.dot(g_bp_arr[0, i, j, i + 1:j + 1],
                           bp_arr[0, i, j, i + 1:j + 1])

                g_w_arr[0, i, j, i + 1:j + 1] += \
                    bp_arr[0, i, j, i + 1:j + 1] \
                    * (g_bp_arr[0, i, j, i + 1:j + 1] - s)

                g_cw_arr[2, i, i + 1:j + 1] += g_w_arr[0, i, j, i + 1:j + 1]
                g_cw_arr[0, i + 1:j + 1, j] += g_w_arr[0, i, j, i + 1:j + 1]

                # right to left incomplete
                g_bp_arr[3, i, j, i:j] += \
                    g_cw_arr[3, i, j] \
                    * w_arr[3, i, j, i:j]

                g_w_arr[3, i, j, i:j] += \
                    g_cw_arr[3, i, j] \
                    * bp_arr[3, i, j, i:j]

                s = np.dot(g_bp_arr[3, i, j, i:j],
                           bp_arr[3, i, j, i:j])

                g_w_arr[3, i, j, i:j] += \
                    bp_arr[3, i, j, i:j] \
                    * (g_bp_arr[3, i, j, i:j] - s)

                g_cw_arr[0, i, i:j] += g_w_arr[3, i, j, i:j]
                g_cw_arr[1, i + 1:j + 1, j] += g_w_arr[3, i, j, i:j]

                # left to right incomplete
                g_bp_arr[2, i, j, i:j] += \
                    g_cw_arr[2, i, j] \
                    * w_arr[2, i, j, i:j]

                g_w_arr[2, i, j, i:j] += \
                    g_cw_arr[2, i, j] \
                    * bp_arr[2, i, j, i:j]

                s = np.dot(g_bp_arr[2, i, j, i:j],
                           bp_arr[2, i, j, i:j])

                g_w_arr[2, i, j, i:j] += \
                    bp_arr[2, i, j, i:j] \
                    * (g_bp_arr[2, i, j, i:j] - s)

                g_cw_arr[0, i, i:j] += g_w_arr[2, i, j, i:j]
                g_cw_arr[1, i + 1:j + 1, j] += g_w_arr[2, i, j, i:j]


if __name__ == '__main__':

    fails_count = 0
    n_tests = 10
    for idx in range(n_tests):
        dim = np.random.randint(low=4, high=12)
        scores = torch.randn((dim, dim), requires_grad=True, dtype=torch.float64)

        print('-----------+-----------+-----------')
        print(f'Test #{idx + 1} (dim = {dim})')
        # test forward pass
        f_test_res = np.array_equal(
                torch.argmax(eisner_surrogate(scores, hard=True), dim=0)[1:],
                parse_proj(scores.detach().numpy())[1:])
        print(f'Forward pass - {"succeeded" if f_test_res else "failed"}')
        # test backward pass
        b_test_res = gradcheck(eisner_surrogate, (scores, False), eps=1e-6, atol=1e-4)
        print(f'Backward pass - {"succeeded" if b_test_res else "failed"}')

        if not f_test_res or not b_test_res:
            fails_count += 1

    print('-----------+-----------+-----------')
    print(f'Summary: {n_tests - fails_count}/{n_tests} successes | {fails_count}/{n_tests} failures')
