{
    logdir: "logdir/glove_run",
    finetunedir: self.logdir + "/finetune",
    model_config: "configs/spider/nl2code-glove.jsonnet",
    model_config_args: {
        att: 0,
        cv_link: true,
        clause_order: null, # strings like "SWGOIF"
        enumerate_order: false,
    },

    eval_name: "glove_run_%s_%d" % [self.eval_use_heuristic, self.eval_beam_size],
    eval_output: self.logdir+"/ie_dirs",
    eval_beam_size: 1,
    eval_use_heuristic: true,
    #eval_steps: [ 1000 * x + 100 for x in std.range(30, 39)] + [40000],
    #eval_steps:[1000*x + 10100 for x in std.range(0,12)],
    eval_steps:[22100],
    eval_section: "val",
}