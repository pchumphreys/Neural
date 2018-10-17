LOG_SIG_CAP_MAX = 2
LOG_SIG_CAP_MIN = -5
LOG_W_CAP_MIN = -10

w_and_mu_and_logsig_t = mlp(
            inputs=self._cond_t_lst,
            layer_sizes=self._layer_sizes,
            output_nonlinearity=None,
        )  # ... x K*Dx*2+K

    w_and_mu_and_logsig_t = tf.reshape(
        w_and_mu_and_logsig_t, shape=(-1, K, 2*Dx+1))

    log_w_t = w_and_mu_and_logsig_t[..., 0]
    mu_t = w_and_mu_and_logsig_t[..., 1:1+Dx]
    log_sig_t = w_and_mu_and_logsig_t[..., 1+Dx:]

    log_sig_t = tf.minimum(log_sig_t, LOG_SIG_CAP_MAX)

    log_w_t = tf.maximum(log_w_t, LOG_W_CAP_MIN)

    return log_w_t, mu_t, log_sig_t

def _create_graph(self):
    Dx = self._Dx

    if len(self._cond_t_lst) > 0:
        N_t = tf.shape(self._cond_t_lst[0])[0]
    else:
        N_t = self._N_pl

    K = self._K

    # Create p(x|z).
    with tf.variable_scope('p'):
        log_ws_t, xz_mus_t, xz_log_sigs_t = self._create_p_xz_params()
        # (N x K), (N x K x Dx), (N x K x Dx)
        xz_sigs_t = tf.exp(xz_log_sigs_t)

        # Sample the latent code.
        z_t = tf.multinomial(logits=log_ws_t, num_samples=1)  # N x 1

        # Choose mixture component corresponding to the latent.
        mask_t = tf.one_hot(
            z_t[:, 0], depth=K, dtype=tf.bool,
            on_value=True, off_value=False
        )
        xz_mu_t = tf.boolean_mask(xz_mus_t, mask_t)  # N x Dx
        xz_sig_t = tf.boolean_mask(xz_sigs_t, mask_t)  # N x Dx

        # Sample x.
        x_t = tf.stop_gradient(xz_mu_t + xz_sig_t * tf.random_normal((N_t, Dx)))  # N x Dx

        # log p(x|z)
        log_p_xz_t = self._create_log_gaussian(
            xz_mus_t, xz_log_sigs_t, x_t[:, None, :]
        )  # N x K

        # log p(x)
        log_p_x_t = tf.reduce_logsumexp(log_p_xz_t + log_ws_t, axis=1)
        log_p_x_t -= tf.reduce_logsumexp(log_ws_t, axis=1)  # N

    reg_loss_t = 0
    reg_loss_t += self._reg * 0.5 * tf.reduce_mean(xz_log_sigs_t ** 2)
    reg_loss_t += self._reg * 0.5 * tf.reduce_mean(xz_mus_t ** 2)

    self._log_p_x_t = log_p_x_t
    self._reg_loss_t = reg_loss_t
    self._x_t = x_t

    self._log_ws_t = log_ws_t
    self._mus_t = xz_mus_t
    self._log_sigs_t = xz_log_sigs_t



def actions_for(self, observations, latents=None,
                name=None, reuse=tf.AUTO_REUSE,
                with_log_pis=False, regularize=False):
    name = name or self.name

    with tf.variable_scope(name, reuse=reuse):
        distribution = GMM(
            K=self._K,
            hidden_layers_sizes=self._hidden_layers,
            Dx=self._Da,
            cond_t_lst=(observations,),
            reg=self._reg
        )

    raw_actions = tf.stop_gradient(distribution.x_t)
    actions = tf.tanh(raw_actions) if self._squash else raw_actions

    # TODO: should always return same shape out
    # Figure out how to make the interface for `log_pis` cleaner
    if with_log_pis:
        # TODO.code_consolidation: should come from log_pis_for
        log_pis = distribution.log_p_t
        if self._squash:
            log_pis -= self._squash_correction(raw_actions)
        return actions, log_pis

    return actions