# Entropy

A person is seating on a bridge above a highway in the USA.
Over a week, this person is counting the cars passing under the bridge.
For each car, this person is putting it in one of 4 buckets: is it a Japanese
car, a Korean car, an American car or some other car.

This person repeats the same experiment but this time in France. Of course
the distribution of cars in France is widely different from the one in the
USA.

At the end of the week, this person is sending the observations in a binary
message: "001110011100011111010100011..." etc.

This person needs to choose an encoding so the length of the message is as
small as possible and the person receiving the message can decode it.

For example, if this person observes 2 American cars, followed by 1 Japanese
car, followed by 2 German cars (other), followed by 1 Japanese car then
finally 1 Korean car, and this person is using as encoding (00 = Japan,
01 = Korea, 10 = USA, 11 = Other), then the message will be "10100011110001"
(read each bit pair one after another: 10, 10, 00, etc.).

Let's illustrate this problem and propose an encoding when this person is in
the USA and then another one when this person is in France - tailored for the
observed distribution of cars.
Let's see how the length of each message is related to the entropy of the
observed distributions.

To run:
```sh
python3 entropy_test.py
```
