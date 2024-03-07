# Summary metrics for ball pushing experiments

In these experiments, data was subsetted to include only the data until the fly brought the ball to the end of the corridor. (a threshold of 40 pixels from the end of the corridor was used to determine when the fly had reached the end of the corridor).

Computed metrics are: 

## NumberEvents

How many interaction events were detected in the video.

## FinalEvent

The event index at which the fly brought the ball at its maximum distance from the start of the corridor.

> Note: This is not necessarily the event at which the fly brought the ball to the end of the corridor; some flies brought the ball only halfway and never went further. In that case, their final event is when the ball first reached halfway.

## FinalTime

The time at which the fly brought the ball at its maximum distance from the start of the corridor.

## SignificantEvents

How many events resulted in a noticeable change in the ball's position (set at 10 px).

## SignificantFirst

The index of the first event that resulted in a noticeable change in the ball's position.

## SignificantFirstTime

The time at which the first event that resulted in a noticeable change in the ball's position occurred.

## Pushes

How many times a significant event resulted in the ball moving toward the end of the corridor.

## Pulls

How many times a significant event resulted in the ball moving away from the end of the corridor.

## PushPullRatio

The ratio of pushes to pulls. (might change this later as the data is massively skewed toward pushes)

## InteractionProportion

The proportion of the data where the fly was interacting with the ball.

## AhaMoment

The time at which the first big push of the ball happened (here I set the threshold at 50 px, roughly 1/4 of the corridor length).

## AhaMomentIndex

The index of the event at which the first big push of the ball happened.

## InsightEffect

The average distance the ball was pushed during a given event after the AhaMoment compared to before the AhaMoment. In this computation the AhaMoment is included in the computation so that if the AhaMoment is also the one leading the ball to the end of the corridor, the InsightEffect is not 0 (as it would be if the AhaMoment was not included).

> Note : This metric is designed to look at how moving the ball for the first time affected flies' behavior to move it further.

## TimeToFinish

The time it took for the fly to bring the ball to the end of the corridor.If the fly never brought the ball to the end of the corridor, this metric is set to the whole video duration.

## SignificantRatio

The proportion of significant events to the total number of events.