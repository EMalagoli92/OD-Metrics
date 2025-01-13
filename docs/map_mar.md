# Mean Average Precision (mAP)

## What is mAP?
**Mean Average Precision (mAP)** is a metric used to evaluate the performance of an object detector.

## How to calculate mAP?
<p align="center">
<img src="../assets/map_1.png" alt="drawing" width="500"/></p>

The starting point is that each prediction of an object detection algorithm is composed by three components:

- the object class $\hat{c} \in \mathbb{N}_{\leq C}^+$, where $C \in \mathbb{N}^+$ is the number of classes;
- the corresponding bounding box $\hat{\textbf{B}} \in \mathbb{R}^4$;
- the confidence score $\hat{s} \in [0, 1]$ showing how confident the detector is about that prediction.

Thus, each detection can be denotaed by:
$$ \hat{\textbf{y}} = ( \hat{c}, \hat{\textbf{B}}, \hat{s} ).$$

Consider a target object to be detected represented by a ground-truth bounding box $\textbf{B} \in \mathbb{R}^4$ and by an object class $c \in \mathbb{N}_{\leq C}^+$:
$$\textbf{y} = ( c, \textbf{B} ). $$
The assessment is done based on:

- a set of $G$ ground-truth:
$$Y = \{(c_i, \textbf{B}_i\}_{i=1,\ldots,G} $$
- a set of $D$ detections:
$$\hat{Y} = \{(\hat{c}_i, \hat{\textbf{B}}_i, \hat{s}_i\}_{i=1,\ldots,D} $$

Now consider a fixed class $\bar{c}$.
If we consider as positive detections only those whose confidence is larger than a confidence threshold $\tau_{s}$, and whose $IoU$ with one ground truth is larger than a $IoU$ threshold $\tau_{IoU}$, thus we can define true positives $TP_{\bar{c}}(\tau_{\text{c}}, \tau_{IoU})$ as:
$$TP_{\bar{c}}(\tau_{s}, \tau_{IoU}) = \{ \hat{\textbf{y}} = ( \hat{c}, \hat{\textbf{B}}, \hat{s} ) \in \hat{Y} \mid \exists \;y = ( c, \textbf{B} ) \in Y: \hat{s} \geq \tau_{s} \wedge \hat{c} = \bar{c} \wedge IoU( \textbf{B}, \hat{\textbf{B}} ) \geq \tau_{IoU} \}. $$
Thus, we can define Precision as:
$$ P_{\bar{c}}(\tau_{s}, \tau_{IoU}) = \frac{TP_{\bar{c}}(\tau_{s}, \tau_{IoU})}{TP_{\bar{c}}(\tau_{s}, \tau_{IoU}) + FP_{\bar{c}}(\tau_{\text{c}}, \tau_{IoU})} = \frac{|TP_{\bar{c}}(\tau_{s}, \tau_{IoU})|}{|\hat{Y}_{\bar{c}, \tau_{s}}|}$$
where $\hat{Y}_{\bar{c}, \tau_{s}} = \{\hat{y} \in \hat{Y} \mid \hat{c} = \bar{c} \wedge \hat{s} \geq \tau_{s}\}$.
Similarly for the Precision
$$ R_{\bar{c}}(\tau_{s}, \tau_{IoU}) = \frac{TP_{\bar{c}}(\tau_{s}, \tau_{IoU})}{TP_{\bar{c}}(\tau_{s}, \tau_{IoU}) + FN_{\bar{c}}(\tau_{s}, \tau_{IoU})} = \frac{|TP_{\bar{c}}(\tau_{\text{c}}, \tau_{IoU})|}{|Y_{\bar{c}}|},$$
where $Y_{\bar{c}} = \{ y \in Y \mid c = \bar{c} \}$.

Now supposed to fix $\bar{\tau}_{IoU}$, the average precision $AP_{\bar{c}}@[\bar{\tau}_{IoU}]$ is a metric based on the area under a $P_{\bar{c}}(\tau_{s}, \bar{\tau}_{IoU})\times R_{\bar{c}}(\tau_{s}, \bar{\tau}_{IoU})$ curve. This area is in practice replaced with a finite sum using certain recall values and different interpolation methods.

One starts by ordering the $K$ different confidence scores output by the detector:

$$T_{\bar{c}} = \{ \tau_{s_k}, k \in \mathbb{N}_{\leq K}^+ \mid \tau_{s_i} > \tau_{s_j} \; \forall i > j\}$$

Interpolated Precision:

$$\tilde{P}_{\bar{c}}(x, \bar{\tau}_{IoU}) = \max_{j \in \mathbb{N}_{\leq K}^+ \mid R_{\bar{c}}(\tau_{s_j}, \bar{\tau}_{IoU}) \geq x} P_{\bar{c}}(\tau_{s_j}, \bar{\tau}_{IoU})$$



$N$-Point Interpolation. In this case the sequence $T_{\bar{c}}$ is chosen such that the corresponding sequence ${R_\bar{c}(\tau_{s_k}, \bar{\tau}_{IoU})}$ is equally spaced in the interval $[0,1]$, that is:

$$R_{\bar{c}}(\tau_{s_k}, \bar{\tau}_{IoU}) = \frac{N - k}{N -1}, \; \; k \in \mathbb{N}^+_{\leq N}$$

and thus:

$$ AP_{\bar{c}}@[\bar{\tau}_{IoU}] = \frac{1}{N} \sum_{k=1}^N  \tilde{P}_{\bar{c}}(\frac{N - k}{N -1}, \bar{\tau}_{IoU}) $$

Popular choices are $N=11$ or $N=101$.

In all-point interpolation:

$$ AP_{\bar{c}}@[\bar{\tau}_{IoU}] = \sum_{k=0}^{K} (R_\bar{c}(\tau_{s_k}, \bar{\tau}_{IoU}) - R_\bar{c}(\tau_{s_{k+1}}, \bar{\tau}_{IoU})) \tilde{P}_{\bar{c}}(R_\bar{c}(\tau_{s_k}, \bar{\tau}_{IoU}), \bar{\tau}_{IoU}) $$

with $\tau_{s_0} = 0$, $R_\bar{c}(\tau_{s_0}, \bar{\tau}_{IoU}) = 1$, $\tau_{s_{K+1}} = 1$, $R_\bar{c}(\tau_{s_{K+1}}, \bar{\tau}_{IoU}) = 0$.

Regardless of the interpolation method, $AP_{\bar{c}}@[\bar{\tau}_{IoU}]$ is obtained individually for each class $\bar{c}$. In large datasets with many classes, it is useful to have a unique metric that is able to represent the exactness of the detections among all classes. For such cases, the mean average precision $mAP@[\bar{\tau}_{IoU}]$ is computed, which is simply:

$$mAP@[\bar{\tau}_{IoU}] = \frac{1}{C}\sum_{c=1}^C AP_{c}@[\bar{\tau}_{IoU}] $$


## Example
Suppose to consider to classes, so $C=2$ and the following $\hat{Y}$.

| Id |  c | IoU  | s  |
|----|---|---|---|
|  a  | 1  | 0.86  | 0.90  |
|  b  |  1 | 0.65  |  0.95 |
|  c  |  1 | 0.44  |  0.7 |
|  d  |  1 | 0.32  |  0.8 |
|  e  |  1 | 0.88  |  0.7 |
|  f  | 2  | 0.48  | 0.82  |
|  g  |  2 | 0.98  |  0.97 |
|  h  |  2 | 0.45  |  0.64 |
|  i  |  2 | 0.67  |  0.63 |
|  l  |  2 | 0.73  |  0.52 |

Now fix the class $\bar{c}=1$ and $\bar{\tau}_{IoU} = 0.5$.
Then:

$$T_{1} = \{ 0.95, 0.90, 0.8, 0.7 \}$$

| Id |  c | IoU  | s  | IoU > 0.5 |
|----|---|---|---|---|
|  b  |  1 | 0.65  |  0.95 | True |
|  a  | 1  | 0.86  | 0.90  | True |
|  d  |  1 | 0.32  |  0.8 | False |
|  c  |  1 | 0.44  |  0.7 | False |
|  e  |  1 | 0.88  |  0.7 | True |

then:

| $\tau_{s}$ |  $Y_{1}$  | $\hat{Y}_{1, \tau_s}$  | $TP_{1}(\tau_s, 0.5)$ | $P_1(\tau_s, 0.5)$ | $R_1(\tau_s, 0.5)$ |
|----|---|---|---|---|---|
| 0.95 | 5 | 1 | 1 | 1.0 | 0.2
| 0.9 | 5 | 2 | 2 | 1.0 | 0.4
| 0.8 | 5 | 3 | 2 | 0.67 | 0.4
| 0.7 | 5 | 5 | 3 | 0.6 | 0.6

So

<div  style="width:1000px;height:600px">            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>                <div id="da71e593-e8de-4708-858a-059610a5b444" class="plotly-graph-div" style="height:100%; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("da71e593-e8de-4708-858a-059610a5b444")) {                    Plotly.newPlot(                        "da71e593-e8de-4708-858a-059610a5b444",                        [{"hovertemplate":"x=%{x}\u003cbr\u003ey=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"","marker":{"color":"rgba(99, 110, 250, 1)","symbol":"circle","size":15},"mode":"markers","name":"","orientation":"v","showlegend":false,"x":[0.2,0.4,0.4,0.6],"xaxis":"x","y":[1.0,1.0,0.67,0.6],"yaxis":"y","type":"scatter"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"$\\Large{R_{1}(\\cdot, 0.5)}$"},"range":[-0.05,1.05],"gridcolor":"rgba(175,178,184,255)"},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"$\\Large{P_{1}(\\cdot, 0.5)}$"},"range":[-0.05,1.05],"autorange":false,"gridcolor":"rgba(175,178,184,255)"},"legend":{"tracegroupgap":0},"margin":{"t":60},"shapes":[{"line":{"color":"rgba(99, 110, 250, 1)","dash":"dash","width":3},"type":"line","x0":0.2,"x1":0.2,"y0":0,"y1":1.0},{"line":{"color":"rgba(99, 110, 250, 1)","dash":"dash","width":3},"type":"line","x0":0.4,"x1":0.4,"y0":0,"y1":1.0},{"line":{"color":"rgba(99, 110, 250, 1)","dash":"dash","width":3},"type":"line","x0":0.4,"x1":0.4,"y0":0,"y1":0.67},{"line":{"color":"rgba(99, 110, 250, 1)","dash":"dash","width":3},"type":"line","x0":0.6,"x1":0.6,"y0":0,"y1":0.6}],"title":{"text":"Precision - Recall curve - Class: 1","y":0.99,"x":0.5,"xanchor":"center","yanchor":"top"},"font":{"family":"Arial","size":15,"color":"rgba(191,192,198,255)"},"plot_bgcolor":"rgba(0, 0, 0, 0)","paper_bgcolor":"rgba(0, 0, 0, 0)"},                        {"responsive": true}                    )                };                            </script>        </div>

## References
