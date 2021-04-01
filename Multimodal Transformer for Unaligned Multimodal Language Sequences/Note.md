## Note

## **Multimodal Transformer for Unaligned Multimodal Language** Sequences



AE+GAE？



## abstract

- natural language
- facial gestures
- acoustic behaviors



Multimodel Transformer(MulT)

end-to-end manner without explicitly aligning the data

heart of our model

**定向成交交叉模型注意力**



### introduction



### related works



#### Human multimodal language analysis



#### Transformer Network



### Proposed Method

在高层，MulT融合了多模型的时间序列，通过前馈fusion，**双向交叉模型transformer**



**Crossmodel attention**

设置有两种模态$\alpha \in R^{T_{\alpha} \times D_{\alpha}}$和$\beta \in R^{T_{\beta}*D_{\beta}}$

- query

  $Q_{\alpha} = X_{\alpha}W$

- key

  $K_{\beta} = X_{\beta}W$

- value

  $V_{\beta} = X_{\beta}W$

