## Course Content of Statistics I, II and III in LinkedIn Learning

### Statistics I

1. What is the starting point to unravel a data story?
    - Look for the **middle** (mean, median and mode)

2. How **spread out** is the data? <br>ALong with "Middle" point, look for `variability`
    - Range: (Max - Min) <br>

        Telling stories with mean and median is still limited. With `Range` it becomes better

        |        | Value |
        |--------|-------|
        | Mean   | 60    |
        | Median | 58    |
        | Range  | 70    |
    
    - `Standard Deviation`:
        -  Approx Definition: Average of all data point's distances from the mean
        - Proper Definition: Square Root of { the mean  of {Square of the difference between each data point and the mean of the data}}

            Std Deviation of the population: <br> &sigma; = &radic;( <sup>&Sigma;(X - &mu;)<sup>2</sup></sup>/ <sub>N</sub> );  

            X  = Value in the data distribution <br>
            &mu; = Mean of the population <br> 
            N = Number of data points

            Std Deviatin of Sample: <br>
            s = &radic; ( <sup> &Sigma;(X - x&#772; )<sup>2</sup> </sup> / <sub> (n-1)</sub> )

        - Why the denominator is `n - 1` in `Sample Std Deviation`? <br>
            - x&#772; is the mean of the sample
            - By empirical evidence (observed in many datasets), <br> &Sigma;<sub>i=1</sub><sup>N</sup> (x<sub>i</sub> - x&#772;)<sup>2</sup> << &Sigma;<sub>i=1</sub><sup>N</sup> (x<sub>i</sub> - &mu;)<sup>2</sup> <br>
            - Hence dividing the sample std deviation by (n-1) makes it "unbiased" and more towards population std deviation <br>
            - Another Explanation: There are only (n-1) degrees of freedom in the calculation of (x<sub>i</sub> - x&#772;)

    - `Z-Score`: 
        - A particular datapoint's distance from the mean measured in standard deviations
             Z-score = ( <sup>X - &mu;</sup> / <sub> &sigma; </sub>)

             = (231 - 139) / 41 = 2.24
             = 231 is 2.24 std deviations from the mean
             = 112 is -0.66 std deviations from the mean 
 
- Interesting points: 
    - Std deviations of two different datasets cannot be compared (e.g.: Salaries of Data Scientists and Consumption of Fuel in cars)  

3. Empirical Rule:
    - Most of the datapoints (68%, 95%, 99.7% ) fall within some std deviations (1,2, and 3 respectively) from the mean
    - In other words, 99.7% of the data that is normally distributed will lie 3 standard deviations from the mean.
    - What is normal distribution? <br>
        The dataset distribution mimics a bell curve
    - Application of the Empirical Rule: 
       - Understanding if a particular data point being an outlier or not 

4. Central Limit Theorem: 
     - Given a population of unknown distribution with mean &mu; and finite variance &sigma;<sup>2</sup>, 
         - If we keep sampling `n` values from the the distribution, and compute sample mean as <br>
         X&#772;<sub>n</sub> ~= ( <sup> X<sub>1</sub> + X<sub>1</sub> + X<sub>n</sub> </sup> / <sub>n</sub>) 
         - As n-> &#8734;, the distribution of the sample means tend to be normal or gaussian (following the bell curve)
    - In simple words, <br>
        - If you have a population with unknown distribution but with a mean of &mu; and std deviation of &sigma; and take sufficiently large number of samples `n` (with replacement), the distribution of means will be approximately normally distributed 
        <br>
        <br>


        ![](https://upload.wikimedia.org/wikipedia/commons/7/7b/IllustrationCentralTheorem.png)



5. Outlier:
    - Outlier is a relative term. There is no absolute definition (like if a datapoint is 2 or 3 &sigma; away from the mean)

    - How to investigate outliers: <br>
    (one should not simply ignore/remove it) 
        - Is this really an outlier?
        - How did this happen?
        - What can we learn?
        - What needs to change (to make it fit into the distribution)?



### Statistics II
    - WIP      