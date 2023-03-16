import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.apis import init_segmentor
from mmcv.cnn.utils import revert_sync_batchnorm
from mmseg.ops import resize


class PSP_OCRNet(nn.Module):
    def __init__(self,psp_config,psp_checkpoint,ocr_config,ocr_checkpoint):
        super(PSP_OCRNet,self).__init__()

        # mode = eval
        self.pspnet=revert_sync_batchnorm(init_segmentor(psp_config,psp_checkpoint,'cpu'))
        self.ocrnet=revert_sync_batchnorm(init_segmentor(ocr_config,ocr_checkpoint,'cpu'))

        # freeze network
        for p1 in self.pspnet.parameters():
            p1.require_grad=False
        for p2 in self.ocrnet.parameters():
            p2.require_grad=False

    def forward(self,x):
        with torch.no_grad():
            # pspnet forward
            psp_feat=self.pspnet.extract_feat(x)
            psp_feat=self.pspnet.decode_head._forward_feature(psp_feat)
            psp_seg=self.pspnet.decode_head.cls_seg(psp_feat)

            # ocrnet forward 
            feats=self.ocrnet.extract_feat(x)
            prev_out=self.ocrnet.decode_head[0].forward_test(feats,None,None)         
            ocr_head=self.ocrnet.decode_head[1]
            feats=ocr_head._transform_inputs(feats)
            feats=ocr_head.bottleneck(feats)
            context = ocr_head.spatial_gather_module(feats, prev_out)
            object_context = ocr_head.object_context_block(feats, context)
            ocr_seg = ocr_head.cls_seg(object_context)

        return {'psp_feat':psp_feat,'ocr_feat':object_context,'psp_seg':psp_seg,'ocr_seg':ocr_seg}



class OCRNet_Segformer(nn.Module):
    def __init__(self,ocr_config,ocr_checkpoint,seg_config,seg_checkpoint):
        super(OCRNet_Segformer,self).__init__()
        self.ocrnet=revert_sync_batchnorm(init_segmentor(ocr_config,ocr_checkpoint,'cpu'))
        self.segformer=revert_sync_batchnorm(init_segmentor(seg_config,seg_checkpoint,'cpu'))

        for p1 in self.ocrnet.parameters():
            p1.require_grad=False
        for p2 in self.segformer.parameters():
            p2.require_grad=False

    def forward(self,x):
        with torch.no_grad():
            # ocrnet forward 
            feats=self.ocrnet.extract_feat(x)
            prev_out=self.ocrnet.decode_head[0].forward_test(feats,None,None)         
            ocr_head=self.ocrnet.decode_head[1]
            feats=ocr_head._transform_inputs(feats)
            feats=ocr_head.bottleneck(feats)
            context = ocr_head.spatial_gather_module(feats, prev_out)
            object_context = ocr_head.object_context_block(feats, context)
            ocr_seg = ocr_head.cls_seg(object_context)
        
            #segformer forward
            inputs=self.segformer.extract_feat(x)
            seg_head=self.segformer.decode_head
            inputs = seg_head._transform_inputs(inputs)
            outs = []
            for idx in range(len(inputs)):
                _x = inputs[idx]
                conv = seg_head.convs[idx]
                outs.append(
                    resize(
                        input=conv(_x),
                        size=inputs[0].shape[2:],
                        mode=seg_head.interpolate_mode,
                        align_corners=seg_head.align_corners))
            seg_feat = seg_head.fusion_conv(torch.cat(outs, dim=1))
            seg_seg = seg_head.cls_seg(seg_feat)

        return {'ocr_feat':object_context,'seg_feat':seg_feat,'ocr_seg':ocr_seg,'seg_seg':seg_seg}

def get_psp_ocrnet(psp_config,psp_checkpoint,ocr_config,ocr_checkpoint):
    return PSP_OCRNet(psp_config,psp_checkpoint,ocr_config,ocr_checkpoint)

def get_ocrnet_segformer(ocr_config,ocr_checkpoint,segformer_config,segformer_checkpoint):
    return OCRNet_Segformer(ocr_config,ocr_checkpoint,segformer_config,segformer_checkpoint)
