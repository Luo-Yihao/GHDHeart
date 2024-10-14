import torch.nn as nn
import torch.nn.functional as F


class BinaryDiceLoss(nn.Module):
	def __init__(self):
		super(BinaryDiceLoss, self).__init__()
	
	def forward(self, input, targets):

		
		# batch size
		N = targets.shape[0]
		# smoothness
		smooth = 1
		# flatten
		input_flat = input.reshape(N, -1)
		targets_flat = targets.reshape(N, -1)
	
		# calculate intersection and union
		intersection = input_flat * targets_flat 
		N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth + 1e-6)
		# mean over batch
		loss = (1 - N_dice_eff).mean()
		return loss
	
class MultiClassDiceLoss(nn.Module):
	def __init__(self, weight=None, ignore_index=[], **kwargs):
		super(MultiClassDiceLoss, self).__init__()
		self.weight = weight
		self.ignore_index = ignore_index
		self.binaryDiceLoss = BinaryDiceLoss()
		self.kwargs = kwargs
	
	def forward(self, input, target):
		"""
			input tesor of shape = (N, C, ...)
			target tensor of shape = (N, C, ...)
		"""
		# target one-hot encoding to (N, C, H, W)
		nclass = input.shape[1]

		assert input.shape == target.shape, "predict & target shape do not match"
		
		
		total_loss = 0
		
		C = nclass - len(self.ignore_index)
		
		# dice loss for each class
		for i in range(nclass):
			if i in self.ignore_index:
				continue
			
			dice_loss = self.binaryDiceLoss(input[:,i,...], target[:,i,...])
			if self.weight is not None:
				assert len(self.weight) == nclass, "Expect weight shape [{}], get[{}]".format(nclass, len(self.weight))
				dice_loss *= self.weight[i]
			total_loss += dice_loss
		
		# mean over classes
		return total_loss / C
