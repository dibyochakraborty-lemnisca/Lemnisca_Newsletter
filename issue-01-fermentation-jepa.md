# your fermentation model is delulu at 10,000L

*lemnisca newsletter, issue 01*
*on scale-up, distribution shift, and why predicting raw signals is cope.*

---

> **the bit**: bioprocess ML doesn't break at scale because the biology changed. it breaks because the *signals* changed and the model never learned anything deeper than "these numbers usually go like this." the actual fix isn't more data, it isn't a chunkier transformer, it isn't another LSTM with attention slapped on top. it's learning a representation of *what state the bioreactor is actually in*. that's the whole pitch behind JEPA, and that's what this issue is about.

---

## 1. the hook

a model that nails yield in a 5L bioreactor often face-plants at 10,000L. same strain. same media. same feed profile copy-pasted from the dev run. and yet the predictions drift, the soft sensor lies, and someone on the MSAT team is staring at a DO trace at 2 a.m. like "why is the validated model off by 30 percent."

the usual move is to collect more data, retune the LSTM, maybe sprinkle in a few engineered features. it almost never works. and there's a reason for that. it's not the model. it's what the model is looking at.

![this is fine](https://api.memegen.link/images/fine/lab_model_at_10000L/this_is_fine.png)

## 2. why the model breaks (and the math nobody puts in slides)

let's actually break this down. a 5L glass vessel and a 10,000L stainless tank are not the same physical system, and the difference is bigger than people give it credit for.

| | lab (5L) | industrial (10,000L) |
|---|---|---|
| mixing time | seconds | up to ~160 s in 100 m³ vessels |
| oxygen | DO is basically uniform | strong O₂ gradients, dead zones, hypoxic pockets |
| substrate | well-mixed | high glucose near the feed port, starvation 4 meters away |
| sensors | dense, fast | sparse, delayed, sometimes one probe per zone |
| CO₂ stripping | easy | limited, local pH crashes follow |

here's the thing. cells respond faster than industrial vessels mix. that's a really important sentence so read it again. the relevant ratio in chemical engineering is the Damköhler number, basically `Da = (rate of reaction or response) / (rate of transport)`. in a 5L vessel, mixing wins, `Da << 1`, the cell sees a homogeneous world. in a 10,000L tank, transport loses, and individual cells get yanked through gradients of glucose, oxygen, and pH on a second-to-minute scale. their transcriptional machinery adapts on a minute-to-hour scale. so the population is permanently in a stress regime that just doesn't exist at 5L.

now translate that into ML. your model was trained under one input distribution `P_train(X)` and is being deployed under a different one `P_deploy(X)`. that's textbook covariate shift. and not the cute kind where you can just reweight the training data. the *meaning* of each variable changes too. a glucose probe reading of 4 g/L at 5L is a description of the entire vessel. the same 4 g/L at 10,000L is a spatial average over a vessel where some cells are drowning in sugar and others are starving. the model has no idea.

so when people say "the model failed," what they actually mean is:

> the model learned how things *looked* in 5L and never learned how the system actually *behaves*. and at scale, the looks change.

![one does not simply](https://api.memegen.link/images/mordor/one_does_not_simply/scale_a_5L_model_to_10000L.png)

## 3. the friction points (where the pain actually lives)

1. **models depend on raw signals.** DO, pH, OUR, CER, OD, viable cell density. every single one shifts in distribution between vessels. this is not a small effect. probe response times alone can be 10x slower at scale.
2. **data is sparse and inconsistent.** industrial runs cost six figures each. you get tens of batches, not millions. cadence differs by site. probes get swapped mid-campaign. one site logs every 30 seconds, another logs every 5 minutes. nobody's going to fix this for you.
3. **labels are delayed and noisy.** titer is HPLC, hours later. productivity is one number per batch. you cannot couple your predictions to ground truth in real time the way you can in vision or NLP, where the label is right there in the next token.
4. **there is no stable notion of process state.** the model never had a concept of "we are now in oxygen-limited late exponential phase with rising acetate." it just had numbers that correlated with yield in the training set.

and that fourth one is the actual bug. everything else is downstream of it.

![distracted boyfriend](https://api.memegen.link/images/db/bioprocess_ml_team/shiny_new_transformer/their_actual_data_problem.png) 

## 4. reframe: stop predicting, start *representing*

the default move is "we need a better predictor." this is mid. if your inputs are unstable across scales, no predictor built on those inputs is going to generalize. you can throw GBDTs, LSTMs, or a 70B-param transformer at it, the input distribution still doesn't match.

the move that actually scales is to ask a different question first.

> *what state is the bioreactor in right now?*

lag phase, mid-exponential, oxygen-limited, acetate-stressed, diauxic shift. these phases exist whether you're at 5L or 10,000L. the *signals describing them* change with vessel geometry. the *phases themselves* don't. they're a property of the cells, not the tank.

so what you want is a model that maps signals to *state*, then predicts in *state* space. that's the jump.

![change my mind](https://api.memegen.link/images/cmm/predicting_raw_signals_at_scale_is_cope.png)

## 5. enter JEPA (chill, no acronym soup)

forget the name for a second. imagine a model that doesn't try to predict the next DO reading or the exact glucose value at minute 47. instead, it tries to answer:

> what kind of situation is the system in, and what kind of situation will it be in 30 minutes from now?

it operates not on numbers, but on a compressed internal description of what's happening. two batches that look numerically nothing alike (one in a 5L glass vessel with 1-second probes, one in a 10,000L stainless tank with sluggish probes and a 90-second mixing dead zone) can map to the *same* internal description if they're in the same biological phase.

that's the idea behind **Joint Embedding Predictive Architectures (JEPA)**, the self-supervised approach Yann LeCun's group at Meta has been pushing (I-JEPA for images, V-JEPA for video). it's not a fermentation idea. it's a general claim about *where* prediction should happen: in embedding space, not in raw-data space.

## 6. how JEPA actually works (the part most blogs skip)

here's the architecture stripped down. three components, one loss.

```
        context window x                       target window y
     (signals up to time t)               (signals around time t+Δ)
              │                                       │
              ▼                                       ▼
        ┌──────────┐                          ┌───────────────┐
        │  encoder  │                          │ target encoder │
        │   f_θ     │                          │     f_ξ       │  (EMA of f_θ,
        └────┬─────┘                          └───────┬───────┘   stop-gradient)
             │                                        │
             ▼                                        ▼
           z_t                                       z_(t+Δ)
             │
             ▼
        ┌─────────────┐
        │ predictor  │
        │    g_φ     │
        └────┬───────┘
             ▼
           ẑ_(t+Δ)  ──────────────►   loss = || ẑ_(t+Δ) − z_(t+Δ) ||²
                                       (in latent space, not signal space)
```

there are three things going on here that matter.

**(a) two encoders, not one.** the context encoder `f_θ` is the one that learns. the target encoder `f_ξ` is an exponential moving average of `f_θ`'s weights, with the gradient blocked. this asymmetry is load-bearing. it's the trick that came out of BYOL and DINO before JEPA inherited it. without it, the model collapses to a constant function (every input maps to the same `z`, loss goes to zero, you've learned nothing).

**(b) the prediction target is the encoder's own output, not the raw data.** this is the actual departure from generative modeling. an autoencoder or a vanilla forecaster tries to reconstruct the future signal. JEPA tries to reconstruct the future *embedding* of that signal. so the model gets full credit for being right about the *gist* of what happens next without having to nail the exact pH at 14:32:07.

**(c) loss is L2 in latent space.** that's it. no contrastive loss, no negatives, no GAN. just "my predicted z should match the target encoder's z." the EMA target plus the predictor asymmetry plus carefully designed masking strategies are what keep the whole thing from collapsing.

*small but important nuance:* in fermentation you'd typically mask spans of the time series, not random points, because the autocorrelation is huge and random masking is too easy. think of it as "hide the next 30 minutes, predict its representation from the previous 2 hours."

for microbes specifically, "meaning" in latent space ends up looking like:

- lag phase
- early exponential growth
- mid-exponential, glucose-replete
- oxygen-limited stress
- acetate inhibition / overflow metabolism
- diauxic shift
- stationary / death

the encoder isn't told these labels. it discovers them because they're the cleanest, most stable axes of variation in the data once you stop forcing the model to memorize sensor quirks.

![buzz](https://api.memegen.link/images/buzz/gradients/gradients_everywhere.png)

## 7. why this fixes scale-up (the actual logical chain)

core claim:

> scale changes signals. scale doesn't change biology.

| thing | changes with scale? | why |
|---|---|---|
| sensor readings | yes | different probes, different sample ports, different averaging |
| spatial gradients | yes | mixing time scales with V^(1/3), reaction time doesn't |
| mass transfer (kLa) | yes | hardware-dependent |
| **biological phases** | no | strain physiology |
| **metabolic regimes** | no | thermodynamics + enzyme kinetics |

a classical model trained on raw signals learns the *top* of that table. it's modeling sensor traces.

a representation-first model learns the *bottom*. its mapping looks like:

```
  5L lab batch    ─► encoder ─► z = "early exponential"
  10,000L batch   ─► encoder ─► z = "early exponential"
```

same `z`. wildly different raw signals. that's literally the definition of generalization.

the formal version is something like: if you can train an encoder `f` such that `f(x_lab) ≈ f(x_indust)` whenever the underlying biological state is the same, then a downstream predictor `h(z) → y` (titer, OUR, whatever) trained on lab data has a real shot at working at 10,000L. you've factored the problem. the messy, scale-dependent part lives in `f`. the clean, scale-invariant part lives in `h`.

this isn't science fiction. a 2024 paper in *Engineering in Life Sciences* on ML in bioreactor scale-up reported that *"an embedding layer improved the capability of artificial neural network models to predict cell growth at large-scale, as this approach captured similarities between the processes"* ([Karimi Alavijeh et al., 2024](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/elsc.202400023)). that's a smaller, less ambitious version of the same idea: stop predicting in signal space, start predicting on top of an embedding. JEPA is the more aggressive form of that bet.

![success kid](https://api.memegen.link/images/success/same_phase_at_5L_and_10000L/same_latent_z.png)

## 8. where ML agents come in (without overselling)

being honest: "learn a good representation" is easy to say and miserable to validate. you'll train ten encoders before you find one whose latent space is actually stable across scales. and you can't tell which one is good by squinting at the loss curve, because every JEPA-style model converges to a small loss number whether it learned anything or not.

this is where ML agents earn their keep. not as oracles. as experimentation infrastructure that runs while you sleep.

concretely, a useful agent loop does four things:

1. **sweep self-supervised pretraining configs.** masking strategy (random vs span vs phase-aware), encoder depth, EMA decay rate, predictor capacity. each combination produces a candidate `f`.
2. **score each `f` on a fixed scale-transfer benchmark.** the metrics that matter are not training loss. they're things like:
   - linear probe accuracy on phase classification (can a logistic regression on `z` recover lag/exp/stationary?)
   - **CKA** (centered kernel alignment) between `f(lab batches)` and `f(industrial batches)` for matched phases
   - downstream titer R² when `h(z) → titer` is trained on lab data and evaluated at industrial scale
3. **detect embedding drift in production.** once a chosen `f` is deployed, monitor the Mahalanobis distance of new batches' latent trajectories from the training cluster centers. when drift exceeds a threshold, alert or trigger retraining. cheap, robust, and catches the failure mode before yield does.
4. **flag when the encoder has nothing to say.** if a new batch lands in a region of latent space that's far from any cluster the encoder ever saw, that's not a prediction, that's an extrapolation, and you should treat it that way.

```python
while True:
    for cfg in encoder_search_space:
        f = pretrain_jepa(cfg, multi_scale_corpus)
        score = (
            0.4 * linear_probe_phase_acc(f)
          + 0.3 * cka_across_scales(f)
          + 0.3 * downstream_titer_r2(f)
        )
    promote(best(score))
    monitor_drift(deployed_encoder)
```

the split is:

> JEPA gives you the right abstraction. agents help you find which version of that abstraction actually holds, and tell you when it stops holding.

no agent makes a bad encoder good. but a good agent loop makes finding the right encoder roughly an order of magnitude faster, and it's the only honest way to monitor a representation in production.

## 9. proof, in three flavors

**(a) adjacent-domain evidence.** representation-first SSL is now the dominant paradigm wherever labels are scarce. I-JEPA: from a single context block, predict the representations of various target blocks in the same image, no pixel reconstruction needed ([Assran et al., CVPR 2023](https://arxiv.org/abs/2301.08243)). V-JEPA scales the same principle to 2M+ unlabeled videos using only a feature-prediction objective ([Meta AI, 2024](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)). BERT-style masked-token pretraining is the entire reason modern NLP works. for time series specifically, recent SSL methods like [PFML (2024)](https://arxiv.org/abs/2411.10087) explicitly target representation learning without collapse, which is the exact failure mode you'd hit if you tried to slap JEPA onto sensor data naively.

**(b) biotech-relevant evidence.** biology is already running this playbook in adjacent layers. protein language models (ESM-style) embed sequences and transfer to dozens of downstream tasks. single-cell omics uses VAEs and contrastive learning to handle batch effects, which is structurally the same problem as scale-up: same biology, different acquisition conditions. a 2024 fermentation soft-sensor study reported a **34% R² improvement and 82% reduction in variability** when augmenting training with VAE-generated synthetic batches ([PMC, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11351132/)). that's a lower-budget version of representation learning helping with exactly the data-sparsity problem fermentation has.

**(c) logical proof.** forget citations. walk the failure mode.

```
  raw-signal model trained on 5L
           │
           ▼
  deployed at 10,000L
           │
           ▼
  signals shift (mixing time, gradients, probe lag)
           │
           ▼
  inputs no longer in P_train
           │
           ▼
  model breaks. no amount of fine-tuning fixes the input distribution.
```

now run it with an encoder.

```
  encoder pretrained on multi-scale signal data
           │
           ▼
  encoder maps 5L and 10,000L into shared latent z
           │
           ▼
  downstream predictor h(z) → y lives in latent space
           │
           ▼
  z is approximately invariant to scale-induced sensor shifts
           │
           ▼
  predictor transfers. as long as the *biological regime* was seen in pretraining.
```

that last clause is doing a lot of work. read on.

![futurama fry](https://api.memegen.link/images/fry/not_sure_if_model_failed/or_biology_changed.png)

## 10. limitations (the part that keeps this honest)

if this newsletter stopped at section 9, you'd be right to be skeptical. JEPA is not magic.

- **it doesn't fix bad data.** mislabeled batches, broken probes, contaminated runs, garbage in, garbage embeddings. SSL learns whatever structure is in the data, including the structure you don't want.
- **new regimes break embeddings.** if 10,000L introduces a phase that *literally never occurs* in your pretraining set (severe local hypoxia in a true dead zone, say) the encoder has nothing to map it to. it'll project the new state into the closest cluster it knows, confidently, and silently. you need at least some multi-scale data in pretraining for the multi-scale claim to mean anything.
- **diversity beats volume.** 50 deliberately varied batches (different feed strategies, induction times, oxygen setpoints) teach the encoder more than 500 nearly-identical ones. this is the opposite of how cell-culture teams usually think about "clean" datasets.
- **representation collapse is real and sneaky.** without the right asymmetric design, the encoder shortcuts to a constant output and your loss looks beautiful. the relevant guardrails (target encoders, predictor capacity, batch statistics regularization) are an active research area, not a solved problem.
- **latent spaces are hard to interpret.** "the encoder says we're in z = [0.31, -0.84, 1.02, ...]" is not a regulatory submission. you need probing classifiers, clustering, and process-knowledge-grounded labels on top before this becomes auditable for QbD or PAT contexts. don't let your CMC team find out from the FDA that you've been deploying an opaque embedding.
- **probes still need to be calibrated.** an encoder that learns from drifting DO probes will learn the drift. PAT hygiene is upstream of all of this.

if you can't accept any of those, you don't want JEPA. you want a pilot plant.

## 11. closing

the problem in bioprocess ML isn't a shortage of models. there are too many already. the problem is a shortage of the right *abstraction layer*.

raw-signal predictors will keep breaking the moment the vessel changes, because they were never modeling the system. they were modeling the signal trace. a soft sensor that lives or dies by probe response time is a soft sensor that lives or dies by hardware.

representation-first methods like JEPA are a serious step toward a model that actually understands what state the bioreactor is in, not just what its sensors are reading right now. they aren't a finished answer. they're the first move that doesn't fall apart at scale on contact.

that's the bet. and at lemnisca, it's the one we're making.

---

### sources

- [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (Assran et al., arXiv:2301.08243)](https://arxiv.org/abs/2301.08243)
- [I-JEPA, Meta AI blog](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/)
- [V-JEPA, Meta AI blog](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)
- [A perspective-driven and technical evaluation of machine learning in bioreactor scale-up (Karimi Alavijeh et al., Eng. Life Sci., 2024)](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/elsc.202400023)
- [Mimicked Mixing-Induced Heterogeneities of Industrial Bioreactors, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10218636/)
- [Substantial gradient mitigation in simulated large-scale bioreactors, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9828524/)
- [Modern Soft-Sensing Modeling Methods for Fermentation Processes, Sensors 2020](https://www.mdpi.com/1424-8220/20/6/1771)
- [Enhancing Fermentation Process Monitoring through Data-Driven Modeling and Synthetic Time Series Generation, PMC 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11351132/)
- [PFML: Self-Supervised Learning of Time-Series Data Without Representation Collapse (arXiv:2411.10087)](https://arxiv.org/abs/2411.10087)

---

*lemnisca. building the AI layer for bioprocess.*
