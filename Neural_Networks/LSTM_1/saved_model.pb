ы
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??
z
Hidden_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(@* 
shared_nameHidden_1/kernel
s
#Hidden_1/kernel/Read/ReadVariableOpReadVariableOpHidden_1/kernel*
_output_shapes

:(@*
dtype0
r
Hidden_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameHidden_1/bias
k
!Hidden_1/bias/Read/ReadVariableOpReadVariableOpHidden_1/bias*
_output_shapes
:@*
dtype0
z
Hidden_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_nameHidden_2/kernel
s
#Hidden_2/kernel/Read/ReadVariableOpReadVariableOpHidden_2/kernel*
_output_shapes

:@@*
dtype0
r
Hidden_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameHidden_2/bias
k
!Hidden_2/bias/Read/ReadVariableOpReadVariableOpHidden_2/bias*
_output_shapes
:@*
dtype0
p

out/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_name
out/kernel
i
out/kernel/Read/ReadVariableOpReadVariableOp
out/kernel*
_output_shapes

:@*
dtype0
h
out/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
out/bias
a
out/bias/Read/ReadVariableOpReadVariableOpout/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/Hidden_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(@*'
shared_nameAdam/Hidden_1/kernel/m
?
*Adam/Hidden_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Hidden_1/kernel/m*
_output_shapes

:(@*
dtype0
?
Adam/Hidden_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/Hidden_1/bias/m
y
(Adam/Hidden_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Hidden_1/bias/m*
_output_shapes
:@*
dtype0
?
Adam/Hidden_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/Hidden_2/kernel/m
?
*Adam/Hidden_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Hidden_2/kernel/m*
_output_shapes

:@@*
dtype0
?
Adam/Hidden_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/Hidden_2/bias/m
y
(Adam/Hidden_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Hidden_2/bias/m*
_output_shapes
:@*
dtype0
~
Adam/out/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*"
shared_nameAdam/out/kernel/m
w
%Adam/out/kernel/m/Read/ReadVariableOpReadVariableOpAdam/out/kernel/m*
_output_shapes

:@*
dtype0
v
Adam/out/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/out/bias/m
o
#Adam/out/bias/m/Read/ReadVariableOpReadVariableOpAdam/out/bias/m*
_output_shapes
:*
dtype0
?
Adam/Hidden_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(@*'
shared_nameAdam/Hidden_1/kernel/v
?
*Adam/Hidden_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Hidden_1/kernel/v*
_output_shapes

:(@*
dtype0
?
Adam/Hidden_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/Hidden_1/bias/v
y
(Adam/Hidden_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Hidden_1/bias/v*
_output_shapes
:@*
dtype0
?
Adam/Hidden_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/Hidden_2/kernel/v
?
*Adam/Hidden_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Hidden_2/kernel/v*
_output_shapes

:@@*
dtype0
?
Adam/Hidden_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/Hidden_2/bias/v
y
(Adam/Hidden_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Hidden_2/bias/v*
_output_shapes
:@*
dtype0
~
Adam/out/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*"
shared_nameAdam/out/kernel/v
w
%Adam/out/kernel/v/Read/ReadVariableOpReadVariableOpAdam/out/kernel/v*
_output_shapes

:@*
dtype0
v
Adam/out/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/out/bias/v
o
#Adam/out/bias/v/Read/ReadVariableOpReadVariableOpAdam/out/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?$
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?$
value?$B?$ B?$
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
h


kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?
iter

beta_1

beta_2
	decay
 learning_rate
m@mAmBmCmDmE
vFvGvHvIvJvK
*

0
1
2
3
4
5
*

0
1
2
3
4
5
 
?
!non_trainable_variables

"layers
#metrics
$layer_regularization_losses
%layer_metrics
	variables
trainable_variables
regularization_losses
 
[Y
VARIABLE_VALUEHidden_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEHidden_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE


0
1


0
1
 
?
&non_trainable_variables

'layers
(metrics
)layer_regularization_losses
*layer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEHidden_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEHidden_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
+non_trainable_variables

,layers
-metrics
.layer_regularization_losses
/layer_metrics
	variables
trainable_variables
regularization_losses
VT
VARIABLE_VALUE
out/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEout/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
0non_trainable_variables

1layers
2metrics
3layer_regularization_losses
4layer_metrics
	variables
trainable_variables
regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2

50
61
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	7total
	8count
9	variables
:	keras_api
D
	;total
	<count
=
_fn_kwargs
>	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

70
81

9	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

;0
<1

>	variables
~|
VARIABLE_VALUEAdam/Hidden_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Hidden_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Hidden_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Hidden_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/out/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/out/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Hidden_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Hidden_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Hidden_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Hidden_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/out/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/out/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
x
serving_default_inputPlaceholder*'
_output_shapes
:?????????(*
dtype0*
shape:?????????(
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputHidden_1/kernelHidden_1/biasHidden_2/kernelHidden_2/bias
out/kernelout/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_142471
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#Hidden_1/kernel/Read/ReadVariableOp!Hidden_1/bias/Read/ReadVariableOp#Hidden_2/kernel/Read/ReadVariableOp!Hidden_2/bias/Read/ReadVariableOpout/kernel/Read/ReadVariableOpout/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/Hidden_1/kernel/m/Read/ReadVariableOp(Adam/Hidden_1/bias/m/Read/ReadVariableOp*Adam/Hidden_2/kernel/m/Read/ReadVariableOp(Adam/Hidden_2/bias/m/Read/ReadVariableOp%Adam/out/kernel/m/Read/ReadVariableOp#Adam/out/bias/m/Read/ReadVariableOp*Adam/Hidden_1/kernel/v/Read/ReadVariableOp(Adam/Hidden_1/bias/v/Read/ReadVariableOp*Adam/Hidden_2/kernel/v/Read/ReadVariableOp(Adam/Hidden_2/bias/v/Read/ReadVariableOp%Adam/out/kernel/v/Read/ReadVariableOp#Adam/out/bias/v/Read/ReadVariableOpConst*(
Tin!
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_142716
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameHidden_1/kernelHidden_1/biasHidden_2/kernelHidden_2/bias
out/kernelout/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/Hidden_1/kernel/mAdam/Hidden_1/bias/mAdam/Hidden_2/kernel/mAdam/Hidden_2/bias/mAdam/out/kernel/mAdam/out/bias/mAdam/Hidden_1/kernel/vAdam/Hidden_1/bias/vAdam/Hidden_2/kernel/vAdam/Hidden_2/bias/vAdam/out/kernel/vAdam/out/bias/v*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_142807??
?
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_142293

inputs!
hidden_1_142254:(@
hidden_1_142256:@!
hidden_2_142271:@@
hidden_2_142273:@

out_142287:@

out_142289:
identity?? Hidden_1/StatefulPartitionedCall? Hidden_2/StatefulPartitionedCall?out/StatefulPartitionedCall?
 Hidden_1/StatefulPartitionedCallStatefulPartitionedCallinputshidden_1_142254hidden_1_142256*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Hidden_1_layer_call_and_return_conditional_losses_142253?
 Hidden_2/StatefulPartitionedCallStatefulPartitionedCall)Hidden_1/StatefulPartitionedCall:output:0hidden_2_142271hidden_2_142273*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Hidden_2_layer_call_and_return_conditional_losses_142270?
out/StatefulPartitionedCallStatefulPartitionedCall)Hidden_2/StatefulPartitionedCall:output:0
out_142287
out_142289*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_out_layer_call_and_return_conditional_losses_142286s
IdentityIdentity$out/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^Hidden_1/StatefulPartitionedCall!^Hidden_2/StatefulPartitionedCall^out/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(: : : : : : 2D
 Hidden_1/StatefulPartitionedCall Hidden_1/StatefulPartitionedCall2D
 Hidden_2/StatefulPartitionedCall Hidden_2/StatefulPartitionedCall2:
out/StatefulPartitionedCallout/StatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
-__inference_sequential_3_layer_call_fn_142505

inputs
unknown:(@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_142376o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_142427	
input!
hidden_1_142411:(@
hidden_1_142413:@!
hidden_2_142416:@@
hidden_2_142418:@

out_142421:@

out_142423:
identity?? Hidden_1/StatefulPartitionedCall? Hidden_2/StatefulPartitionedCall?out/StatefulPartitionedCall?
 Hidden_1/StatefulPartitionedCallStatefulPartitionedCallinputhidden_1_142411hidden_1_142413*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Hidden_1_layer_call_and_return_conditional_losses_142253?
 Hidden_2/StatefulPartitionedCallStatefulPartitionedCall)Hidden_1/StatefulPartitionedCall:output:0hidden_2_142416hidden_2_142418*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Hidden_2_layer_call_and_return_conditional_losses_142270?
out/StatefulPartitionedCallStatefulPartitionedCall)Hidden_2/StatefulPartitionedCall:output:0
out_142421
out_142423*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_out_layer_call_and_return_conditional_losses_142286s
IdentityIdentity$out/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^Hidden_1/StatefulPartitionedCall!^Hidden_2/StatefulPartitionedCall^out/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(: : : : : : 2D
 Hidden_1/StatefulPartitionedCall Hidden_1/StatefulPartitionedCall2D
 Hidden_2/StatefulPartitionedCall Hidden_2/StatefulPartitionedCall2:
out/StatefulPartitionedCallout/StatefulPartitionedCall:N J
'
_output_shapes
:?????????(

_user_specified_nameinput
?
?
-__inference_sequential_3_layer_call_fn_142488

inputs
unknown:(@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_142293o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
-__inference_sequential_3_layer_call_fn_142308	
input
unknown:(@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_142293o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????(

_user_specified_nameinput
?	
?
?__inference_out_layer_call_and_return_conditional_losses_142286

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
)__inference_Hidden_1_layer_call_fn_142562

inputs
unknown:(@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Hidden_1_layer_call_and_return_conditional_losses_142253o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_142529

inputs9
'hidden_1_matmul_readvariableop_resource:(@6
(hidden_1_biasadd_readvariableop_resource:@9
'hidden_2_matmul_readvariableop_resource:@@6
(hidden_2_biasadd_readvariableop_resource:@4
"out_matmul_readvariableop_resource:@1
#out_biasadd_readvariableop_resource:
identity??Hidden_1/BiasAdd/ReadVariableOp?Hidden_1/MatMul/ReadVariableOp?Hidden_2/BiasAdd/ReadVariableOp?Hidden_2/MatMul/ReadVariableOp?out/BiasAdd/ReadVariableOp?out/MatMul/ReadVariableOp?
Hidden_1/MatMul/ReadVariableOpReadVariableOp'hidden_1_matmul_readvariableop_resource*
_output_shapes

:(@*
dtype0{
Hidden_1/MatMulMatMulinputs&Hidden_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Hidden_1/BiasAdd/ReadVariableOpReadVariableOp(hidden_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
Hidden_1/BiasAddBiasAddHidden_1/MatMul:product:0'Hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@h
Hidden_1/SigmoidSigmoidHidden_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
Hidden_2/MatMul/ReadVariableOpReadVariableOp'hidden_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
Hidden_2/MatMulMatMulHidden_1/Sigmoid:y:0&Hidden_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Hidden_2/BiasAdd/ReadVariableOpReadVariableOp(hidden_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
Hidden_2/BiasAddBiasAddHidden_2/MatMul:product:0'Hidden_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@h
Hidden_2/SigmoidSigmoidHidden_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@|
out/MatMul/ReadVariableOpReadVariableOp"out_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0

out/MatMulMatMulHidden_2/Sigmoid:y:0!out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????z
out/BiasAdd/ReadVariableOpReadVariableOp#out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
out/BiasAddBiasAddout/MatMul:product:0"out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????c
IdentityIdentityout/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^Hidden_1/BiasAdd/ReadVariableOp^Hidden_1/MatMul/ReadVariableOp ^Hidden_2/BiasAdd/ReadVariableOp^Hidden_2/MatMul/ReadVariableOp^out/BiasAdd/ReadVariableOp^out/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(: : : : : : 2B
Hidden_1/BiasAdd/ReadVariableOpHidden_1/BiasAdd/ReadVariableOp2@
Hidden_1/MatMul/ReadVariableOpHidden_1/MatMul/ReadVariableOp2B
Hidden_2/BiasAdd/ReadVariableOpHidden_2/BiasAdd/ReadVariableOp2@
Hidden_2/MatMul/ReadVariableOpHidden_2/MatMul/ReadVariableOp28
out/BiasAdd/ReadVariableOpout/BiasAdd/ReadVariableOp26
out/MatMul/ReadVariableOpout/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
$__inference_out_layer_call_fn_142602

inputs
unknown:@
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_out_layer_call_and_return_conditional_losses_142286o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
?__inference_out_layer_call_and_return_conditional_losses_142612

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_142553

inputs9
'hidden_1_matmul_readvariableop_resource:(@6
(hidden_1_biasadd_readvariableop_resource:@9
'hidden_2_matmul_readvariableop_resource:@@6
(hidden_2_biasadd_readvariableop_resource:@4
"out_matmul_readvariableop_resource:@1
#out_biasadd_readvariableop_resource:
identity??Hidden_1/BiasAdd/ReadVariableOp?Hidden_1/MatMul/ReadVariableOp?Hidden_2/BiasAdd/ReadVariableOp?Hidden_2/MatMul/ReadVariableOp?out/BiasAdd/ReadVariableOp?out/MatMul/ReadVariableOp?
Hidden_1/MatMul/ReadVariableOpReadVariableOp'hidden_1_matmul_readvariableop_resource*
_output_shapes

:(@*
dtype0{
Hidden_1/MatMulMatMulinputs&Hidden_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Hidden_1/BiasAdd/ReadVariableOpReadVariableOp(hidden_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
Hidden_1/BiasAddBiasAddHidden_1/MatMul:product:0'Hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@h
Hidden_1/SigmoidSigmoidHidden_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
Hidden_2/MatMul/ReadVariableOpReadVariableOp'hidden_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
Hidden_2/MatMulMatMulHidden_1/Sigmoid:y:0&Hidden_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
Hidden_2/BiasAdd/ReadVariableOpReadVariableOp(hidden_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
Hidden_2/BiasAddBiasAddHidden_2/MatMul:product:0'Hidden_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@h
Hidden_2/SigmoidSigmoidHidden_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@|
out/MatMul/ReadVariableOpReadVariableOp"out_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0

out/MatMulMatMulHidden_2/Sigmoid:y:0!out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????z
out/BiasAdd/ReadVariableOpReadVariableOp#out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
out/BiasAddBiasAddout/MatMul:product:0"out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????c
IdentityIdentityout/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^Hidden_1/BiasAdd/ReadVariableOp^Hidden_1/MatMul/ReadVariableOp ^Hidden_2/BiasAdd/ReadVariableOp^Hidden_2/MatMul/ReadVariableOp^out/BiasAdd/ReadVariableOp^out/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(: : : : : : 2B
Hidden_1/BiasAdd/ReadVariableOpHidden_1/BiasAdd/ReadVariableOp2@
Hidden_1/MatMul/ReadVariableOpHidden_1/MatMul/ReadVariableOp2B
Hidden_2/BiasAdd/ReadVariableOpHidden_2/BiasAdd/ReadVariableOp2@
Hidden_2/MatMul/ReadVariableOpHidden_2/MatMul/ReadVariableOp28
out/BiasAdd/ReadVariableOpout/BiasAdd/ReadVariableOp26
out/MatMul/ReadVariableOpout/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
)__inference_Hidden_2_layer_call_fn_142582

inputs
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Hidden_2_layer_call_and_return_conditional_losses_142270o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?;
?

__inference__traced_save_142716
file_prefix.
*savev2_hidden_1_kernel_read_readvariableop,
(savev2_hidden_1_bias_read_readvariableop.
*savev2_hidden_2_kernel_read_readvariableop,
(savev2_hidden_2_bias_read_readvariableop)
%savev2_out_kernel_read_readvariableop'
#savev2_out_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_hidden_1_kernel_m_read_readvariableop3
/savev2_adam_hidden_1_bias_m_read_readvariableop5
1savev2_adam_hidden_2_kernel_m_read_readvariableop3
/savev2_adam_hidden_2_bias_m_read_readvariableop0
,savev2_adam_out_kernel_m_read_readvariableop.
*savev2_adam_out_bias_m_read_readvariableop5
1savev2_adam_hidden_1_kernel_v_read_readvariableop3
/savev2_adam_hidden_1_bias_v_read_readvariableop5
1savev2_adam_hidden_2_kernel_v_read_readvariableop3
/savev2_adam_hidden_2_bias_v_read_readvariableop0
,savev2_adam_out_kernel_v_read_readvariableop.
*savev2_adam_out_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_hidden_1_kernel_read_readvariableop(savev2_hidden_1_bias_read_readvariableop*savev2_hidden_2_kernel_read_readvariableop(savev2_hidden_2_bias_read_readvariableop%savev2_out_kernel_read_readvariableop#savev2_out_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_hidden_1_kernel_m_read_readvariableop/savev2_adam_hidden_1_bias_m_read_readvariableop1savev2_adam_hidden_2_kernel_m_read_readvariableop/savev2_adam_hidden_2_bias_m_read_readvariableop,savev2_adam_out_kernel_m_read_readvariableop*savev2_adam_out_bias_m_read_readvariableop1savev2_adam_hidden_1_kernel_v_read_readvariableop/savev2_adam_hidden_1_bias_v_read_readvariableop1savev2_adam_hidden_2_kernel_v_read_readvariableop/savev2_adam_hidden_2_bias_v_read_readvariableop,savev2_adam_out_kernel_v_read_readvariableop*savev2_adam_out_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 **
dtypes 
2	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :(@:@:@@:@:@:: : : : : : : : : :(@:@:@@:@:@::(@:@:@@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:(@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:(@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:(@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: 
? 
?
!__inference__wrapped_model_142235	
inputF
4sequential_3_hidden_1_matmul_readvariableop_resource:(@C
5sequential_3_hidden_1_biasadd_readvariableop_resource:@F
4sequential_3_hidden_2_matmul_readvariableop_resource:@@C
5sequential_3_hidden_2_biasadd_readvariableop_resource:@A
/sequential_3_out_matmul_readvariableop_resource:@>
0sequential_3_out_biasadd_readvariableop_resource:
identity??,sequential_3/Hidden_1/BiasAdd/ReadVariableOp?+sequential_3/Hidden_1/MatMul/ReadVariableOp?,sequential_3/Hidden_2/BiasAdd/ReadVariableOp?+sequential_3/Hidden_2/MatMul/ReadVariableOp?'sequential_3/out/BiasAdd/ReadVariableOp?&sequential_3/out/MatMul/ReadVariableOp?
+sequential_3/Hidden_1/MatMul/ReadVariableOpReadVariableOp4sequential_3_hidden_1_matmul_readvariableop_resource*
_output_shapes

:(@*
dtype0?
sequential_3/Hidden_1/MatMulMatMulinput3sequential_3/Hidden_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
,sequential_3/Hidden_1/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_hidden_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_3/Hidden_1/BiasAddBiasAdd&sequential_3/Hidden_1/MatMul:product:04sequential_3/Hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
sequential_3/Hidden_1/SigmoidSigmoid&sequential_3/Hidden_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
+sequential_3/Hidden_2/MatMul/ReadVariableOpReadVariableOp4sequential_3_hidden_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
sequential_3/Hidden_2/MatMulMatMul!sequential_3/Hidden_1/Sigmoid:y:03sequential_3/Hidden_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
,sequential_3/Hidden_2/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_hidden_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_3/Hidden_2/BiasAddBiasAdd&sequential_3/Hidden_2/MatMul:product:04sequential_3/Hidden_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
sequential_3/Hidden_2/SigmoidSigmoid&sequential_3/Hidden_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
&sequential_3/out/MatMul/ReadVariableOpReadVariableOp/sequential_3_out_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
sequential_3/out/MatMulMatMul!sequential_3/Hidden_2/Sigmoid:y:0.sequential_3/out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
'sequential_3/out/BiasAdd/ReadVariableOpReadVariableOp0sequential_3_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_3/out/BiasAddBiasAdd!sequential_3/out/MatMul:product:0/sequential_3/out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p
IdentityIdentity!sequential_3/out/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp-^sequential_3/Hidden_1/BiasAdd/ReadVariableOp,^sequential_3/Hidden_1/MatMul/ReadVariableOp-^sequential_3/Hidden_2/BiasAdd/ReadVariableOp,^sequential_3/Hidden_2/MatMul/ReadVariableOp(^sequential_3/out/BiasAdd/ReadVariableOp'^sequential_3/out/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(: : : : : : 2\
,sequential_3/Hidden_1/BiasAdd/ReadVariableOp,sequential_3/Hidden_1/BiasAdd/ReadVariableOp2Z
+sequential_3/Hidden_1/MatMul/ReadVariableOp+sequential_3/Hidden_1/MatMul/ReadVariableOp2\
,sequential_3/Hidden_2/BiasAdd/ReadVariableOp,sequential_3/Hidden_2/BiasAdd/ReadVariableOp2Z
+sequential_3/Hidden_2/MatMul/ReadVariableOp+sequential_3/Hidden_2/MatMul/ReadVariableOp2R
'sequential_3/out/BiasAdd/ReadVariableOp'sequential_3/out/BiasAdd/ReadVariableOp2P
&sequential_3/out/MatMul/ReadVariableOp&sequential_3/out/MatMul/ReadVariableOp:N J
'
_output_shapes
:?????????(

_user_specified_nameinput
?
?
$__inference_signature_wrapper_142471	
input
unknown:(@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_142235o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????(

_user_specified_nameinput
?

?
D__inference_Hidden_2_layer_call_and_return_conditional_losses_142593

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????@Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_142446	
input!
hidden_1_142430:(@
hidden_1_142432:@!
hidden_2_142435:@@
hidden_2_142437:@

out_142440:@

out_142442:
identity?? Hidden_1/StatefulPartitionedCall? Hidden_2/StatefulPartitionedCall?out/StatefulPartitionedCall?
 Hidden_1/StatefulPartitionedCallStatefulPartitionedCallinputhidden_1_142430hidden_1_142432*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Hidden_1_layer_call_and_return_conditional_losses_142253?
 Hidden_2/StatefulPartitionedCallStatefulPartitionedCall)Hidden_1/StatefulPartitionedCall:output:0hidden_2_142435hidden_2_142437*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Hidden_2_layer_call_and_return_conditional_losses_142270?
out/StatefulPartitionedCallStatefulPartitionedCall)Hidden_2/StatefulPartitionedCall:output:0
out_142440
out_142442*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_out_layer_call_and_return_conditional_losses_142286s
IdentityIdentity$out/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^Hidden_1/StatefulPartitionedCall!^Hidden_2/StatefulPartitionedCall^out/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(: : : : : : 2D
 Hidden_1/StatefulPartitionedCall Hidden_1/StatefulPartitionedCall2D
 Hidden_2/StatefulPartitionedCall Hidden_2/StatefulPartitionedCall2:
out/StatefulPartitionedCallout/StatefulPartitionedCall:N J
'
_output_shapes
:?????????(

_user_specified_nameinput
?

?
D__inference_Hidden_1_layer_call_and_return_conditional_losses_142253

inputs0
matmul_readvariableop_resource:(@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????@Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?

?
D__inference_Hidden_2_layer_call_and_return_conditional_losses_142270

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????@Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
D__inference_Hidden_1_layer_call_and_return_conditional_losses_142573

inputs0
matmul_readvariableop_resource:(@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????@Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?
?
H__inference_sequential_3_layer_call_and_return_conditional_losses_142376

inputs!
hidden_1_142360:(@
hidden_1_142362:@!
hidden_2_142365:@@
hidden_2_142367:@

out_142370:@

out_142372:
identity?? Hidden_1/StatefulPartitionedCall? Hidden_2/StatefulPartitionedCall?out/StatefulPartitionedCall?
 Hidden_1/StatefulPartitionedCallStatefulPartitionedCallinputshidden_1_142360hidden_1_142362*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Hidden_1_layer_call_and_return_conditional_losses_142253?
 Hidden_2/StatefulPartitionedCallStatefulPartitionedCall)Hidden_1/StatefulPartitionedCall:output:0hidden_2_142365hidden_2_142367*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Hidden_2_layer_call_and_return_conditional_losses_142270?
out/StatefulPartitionedCallStatefulPartitionedCall)Hidden_2/StatefulPartitionedCall:output:0
out_142370
out_142372*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_out_layer_call_and_return_conditional_losses_142286s
IdentityIdentity$out/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp!^Hidden_1/StatefulPartitionedCall!^Hidden_2/StatefulPartitionedCall^out/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(: : : : : : 2D
 Hidden_1/StatefulPartitionedCall Hidden_1/StatefulPartitionedCall2D
 Hidden_2/StatefulPartitionedCall Hidden_2/StatefulPartitionedCall2:
out/StatefulPartitionedCallout/StatefulPartitionedCall:O K
'
_output_shapes
:?????????(
 
_user_specified_nameinputs
?l
?
"__inference__traced_restore_142807
file_prefix2
 assignvariableop_hidden_1_kernel:(@.
 assignvariableop_1_hidden_1_bias:@4
"assignvariableop_2_hidden_2_kernel:@@.
 assignvariableop_3_hidden_2_bias:@/
assignvariableop_4_out_kernel:@)
assignvariableop_5_out_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: <
*assignvariableop_15_adam_hidden_1_kernel_m:(@6
(assignvariableop_16_adam_hidden_1_bias_m:@<
*assignvariableop_17_adam_hidden_2_kernel_m:@@6
(assignvariableop_18_adam_hidden_2_bias_m:@7
%assignvariableop_19_adam_out_kernel_m:@1
#assignvariableop_20_adam_out_bias_m:<
*assignvariableop_21_adam_hidden_1_kernel_v:(@6
(assignvariableop_22_adam_hidden_1_bias_v:@<
*assignvariableop_23_adam_hidden_2_kernel_v:@@6
(assignvariableop_24_adam_hidden_2_bias_v:@7
%assignvariableop_25_adam_out_kernel_v:@1
#assignvariableop_26_adam_out_bias_v:
identity_28??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp assignvariableop_hidden_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp assignvariableop_1_hidden_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_hidden_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp assignvariableop_3_hidden_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_out_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_out_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_hidden_1_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_hidden_1_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_hidden_2_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_hidden_2_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp%assignvariableop_19_adam_out_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp#assignvariableop_20_adam_out_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_hidden_1_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_hidden_1_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_hidden_2_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_hidden_2_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp%assignvariableop_25_adam_out_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp#assignvariableop_26_adam_out_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_28IdentityIdentity_27:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_28Identity_28:output:0*K
_input_shapes:
8: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
-__inference_sequential_3_layer_call_fn_142408	
input
unknown:(@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_3_layer_call_and_return_conditional_losses_142376o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:?????????(

_user_specified_nameinput"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
7
input.
serving_default_input:0?????????(7
out0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?N
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
L__call__
*M&call_and_return_all_conditional_losses
N_default_save_signature"
_tf_keras_sequential
?


kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
O__call__
*P&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
S__call__
*T&call_and_return_all_conditional_losses"
_tf_keras_layer
?
iter

beta_1

beta_2
	decay
 learning_rate
m@mAmBmCmDmE
vFvGvHvIvJvK"
	optimizer
J

0
1
2
3
4
5"
trackable_list_wrapper
J

0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
!non_trainable_variables

"layers
#metrics
$layer_regularization_losses
%layer_metrics
	variables
trainable_variables
regularization_losses
L__call__
N_default_save_signature
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
,
Userving_default"
signature_map
!:(@2Hidden_1/kernel
:@2Hidden_1/bias
.

0
1"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
&non_trainable_variables

'layers
(metrics
)layer_regularization_losses
*layer_metrics
	variables
trainable_variables
regularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
!:@@2Hidden_2/kernel
:@2Hidden_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
+non_trainable_variables

,layers
-metrics
.layer_regularization_losses
/layer_metrics
	variables
trainable_variables
regularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
:@2
out/kernel
:2out/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0non_trainable_variables

1layers
2metrics
3layer_regularization_losses
4layer_metrics
	variables
trainable_variables
regularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	7total
	8count
9	variables
:	keras_api"
_tf_keras_metric
^
	;total
	<count
=
_fn_kwargs
>	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
70
81"
trackable_list_wrapper
-
9	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
;0
<1"
trackable_list_wrapper
-
>	variables"
_generic_user_object
&:$(@2Adam/Hidden_1/kernel/m
 :@2Adam/Hidden_1/bias/m
&:$@@2Adam/Hidden_2/kernel/m
 :@2Adam/Hidden_2/bias/m
!:@2Adam/out/kernel/m
:2Adam/out/bias/m
&:$(@2Adam/Hidden_1/kernel/v
 :@2Adam/Hidden_1/bias/v
&:$@@2Adam/Hidden_2/kernel/v
 :@2Adam/Hidden_2/bias/v
!:@2Adam/out/kernel/v
:2Adam/out/bias/v
?2?
-__inference_sequential_3_layer_call_fn_142308
-__inference_sequential_3_layer_call_fn_142488
-__inference_sequential_3_layer_call_fn_142505
-__inference_sequential_3_layer_call_fn_142408?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_sequential_3_layer_call_and_return_conditional_losses_142529
H__inference_sequential_3_layer_call_and_return_conditional_losses_142553
H__inference_sequential_3_layer_call_and_return_conditional_losses_142427
H__inference_sequential_3_layer_call_and_return_conditional_losses_142446?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_142235input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_Hidden_1_layer_call_fn_142562?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_Hidden_1_layer_call_and_return_conditional_losses_142573?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_Hidden_2_layer_call_fn_142582?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_Hidden_2_layer_call_and_return_conditional_losses_142593?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_out_layer_call_fn_142602?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_out_layer_call_and_return_conditional_losses_142612?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_142471input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
D__inference_Hidden_1_layer_call_and_return_conditional_losses_142573\
/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????@
? |
)__inference_Hidden_1_layer_call_fn_142562O
/?,
%?"
 ?
inputs?????????(
? "??????????@?
D__inference_Hidden_2_layer_call_and_return_conditional_losses_142593\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? |
)__inference_Hidden_2_layer_call_fn_142582O/?,
%?"
 ?
inputs?????????@
? "??????????@?
!__inference__wrapped_model_142235c
.?+
$?!
?
input?????????(
? ")?&
$
out?
out??????????
?__inference_out_layer_call_and_return_conditional_losses_142612\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? w
$__inference_out_layer_call_fn_142602O/?,
%?"
 ?
inputs?????????@
? "???????????
H__inference_sequential_3_layer_call_and_return_conditional_losses_142427g
6?3
,?)
?
input?????????(
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_3_layer_call_and_return_conditional_losses_142446g
6?3
,?)
?
input?????????(
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_3_layer_call_and_return_conditional_losses_142529h
7?4
-?*
 ?
inputs?????????(
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_3_layer_call_and_return_conditional_losses_142553h
7?4
-?*
 ?
inputs?????????(
p

 
? "%?"
?
0?????????
? ?
-__inference_sequential_3_layer_call_fn_142308Z
6?3
,?)
?
input?????????(
p 

 
? "???????????
-__inference_sequential_3_layer_call_fn_142408Z
6?3
,?)
?
input?????????(
p

 
? "???????????
-__inference_sequential_3_layer_call_fn_142488[
7?4
-?*
 ?
inputs?????????(
p 

 
? "???????????
-__inference_sequential_3_layer_call_fn_142505[
7?4
-?*
 ?
inputs?????????(
p

 
? "???????????
$__inference_signature_wrapper_142471l
7?4
? 
-?*
(
input?
input?????????(")?&
$
out?
out?????????