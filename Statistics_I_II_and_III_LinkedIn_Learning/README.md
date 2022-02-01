## Probability & Statistics 
### Notes from LinkedIn Learning Course

### <u>Table of Contents</u>

<details><summary> Course 1 - Probability Foundation </summary>

 <br>
    
- Introduction to Prob:
    - The essential understanding of Permutations, Combinations and Percentiles
- Multiple Event Probability: 
    - The Addition Rule of Probability
    - The Multiplication Rule of Probability
    - Conditional Probability
    - Probability Trees
    - Bayes Theorem 
- Discrete and Continious Probabilities
    
</details>


<details><summary> Course 2 - Statistics Foundation </summary>

- Charts and Tables (not discussed)
- Middle (mean, median and mode)
- How **Spread Out** is the data
    - Range
    - Standard Deviation
    - Z-Score
- Empirical Rule
- Central Limit Theorem (just intro and an article)
- Outlier Treatment (Key Qs to Ask)
</details>

<br>


### Probability Foundation Notes

<details><summary> Key Points </summary>

<br>

1. **Quick Introduction to Probability**: <br>
- Probability - Odds of a particular event happening over all possible outcomes 
- Prob = <sup> # of Desired Outcomes </sup> / <sub> Total # of Possible Outcomes </sub>
- Basic types of Probability: <br>
    - Classical Probability
    - Empirical Probability
    - Subjective Probability

- Both Classical and Empirical Probabilities are Objective Probabilities where
    - in Classical: The odds won't change. They are based on `formal reasoning`. They based on established events/theory. 
        E.g.: flipping a fair coin, picking a card from a usual pack of 52 cards
    - in Emprical: The odds are based on `experimental or historical data`
        E.g.: What is the chance of a particular player scoring above 50 runs this match?
        (this can be determined by the historical data of that player)
- Subjective probabilities are based on `personal beliefs`

- Types of odds: 
    - Even Odds | Equally likely events
        - E.g.: Flipping a coin or Rolling a fair die
    - Weighted Odds | Events with unequal chances of occuring 
        - E.g.: Chances of occurrence of rain in Chennai today

2. How to **count number of possible outcomes**?
- `Permutations`: If interested in the order of things
    - E.g.: What is the chance for Students A and B to win top 2 prizes of a competition from a class of 10; There are 8 contestants and 3 prizes. How many different possible outcomes are there?
    - nPk = <sup>n!</sup> / <sub>(n-k)!</sub>

- `Combinations`: If order is not important
    - E.g.: What are the number of ways you can pick 4 members from a team of 12 members?
    - nCk = <sup>n!</sup> / <sub>k!*(n-k)!</sub>

- Q1) There are 8 contestants and 3 prizes. How many different possible outcomes are there?
    - "Permutations" problem because **order** is important
    - 8! / (8-3)! = 8 X 7 X 6 = 336

- Q2A) In how many ways can you "pick" 4 member team from a total 12 members?
     - "Combinations" problem
     - 12! / 4! * (12-4)! = 495

- Q2B) In how many of these 495 combinations, do sisters Layla and Olivia join the same team?
     - Assuming the sisters have already been inducted into the team,there are 
         -`10! / 2! * (10-2)!=45` ways to pick the remaining two members  
 
- `Percentile Rank` = Percentile rank of a given score is the percentage of scores in its freq distribution that are less than that score

  ```
    PR =( CF' + (0.5 X F) ) / N * 100
		CF' = Cumulative Freq (excluding the current score) = Count of all scores less than the score of interest
		F  = Freq of the score of interest 
        N = Total number of scores in the distribution
    
  ```
  3. **Multiple Event Probabilities**: <br>
  	
		E.g. of single event probabilities: <br>
        Heads or Tails: <br>
		    - Heads or Tails <br>
		    - Rain or No Rain <br>
		
        E.g. of Multiple event probabilities: <br>
		*Sports Example*: <br>
			- 30% chance of a player scoring a goal <br>
			- 40% chance of that player's team winning <br>
		
		Q) Is there a relationship between the team winning and this player's success in scoring? <br>
		
		*Healthcare example*: <br>
			- 1/ 10K people gets a particular rare disease <br>
			- Test is accurate only 98% of the time <br>

        Q) What are the chances a person who is tested positive is actually false positive? <br>
 
        *Employment example*: <br>
			- Only 4 out of 20 get an interview <br>
			- What is the prob that friends Mohan and Lily both get a slot in the interview? <br>
            - What if Mohan gets one of the 4 slots, what is the prob of Lily securing one of the other 3 slots? <br>

   		
	More Probability Tools: <br>
		- Conditional Probability <br>
		- Dependent vs Independent events <br>
        - Probability Trees and Bayes Theorem (both useful in managing multiple event scenarios) <br>

 4. More Probability Questions (using above concepts): <br>
   
    ```
    Q1. there are 6 people getting rewarded, what are the chances that 2 people - X and Y  - win the gift?
	
	Total number of outcomes = 
	6! / (4! * 2!) = 15 
	
	Number of outcomes where X, Y are both winning = [(X,Y), (Y,X)]
	Order does not matter, hence number of outcomes of X and Y winning = 1
    = 1/15 = 0.0667 

    ```
    
    ```
    Q2.What is the probability of rolling two dice with each die throwing 1?
    
    1/6 * 1/6 = 0.0278 = 0.0278 2.7%
    ```
    
    ```
    Q3. There are 10 cards with 3 of them having X on them. What are the odds that 2 cards picked at random have X on them?

    Solution method 1: `Combinations` approach
	Total number of combinations = 10C2 
	= 10! /(8! x 2!) = 45 
	
	Total number of combinations with 2 X on them = 3C2 = 3! / (2! * 1!) = 3 
	
	Prob of picking 2 cards at random  where both are X = 3/45 = 0.0667 
	
	
	Solution method 2: `Conditional Prob` approach
	
	Chances of picking X card in attempt 1 = 3/10 = 0.3 
	
	Chances of picking an X card in attempt 2 as well = 2/9 = 0.2222 
	
	Chance of picking 2 X cards (2 dependent events) = 0.3 * 0.2222 = 0.0667   
    ```
    
    ```
    Q4. What are the chances that a medical test taken was false positive?
    (it is a conditional probability where Given that the result is positive, what are the chances that it is false)

    Disease or No Disease
    Positive or Negative Test Result (could be false positive or true negative also)

    Stats:
        1. only 1 in 10,000 has disease
        2. those with disease test positive 99% of time (that remaining 1% is False Negative or Type II)
        3. 2% of healthy paitents will test positive


    Tree 
        - Stage 1:
        Disease 1/10000 = 0.0001 
            - Stage 2:
            Positive: 99/100 = 0.99 
            Negative: 1/100 = 0.01 

        No Disease 9999/10000 = 0.9999 
            Positive: 2/100 = 0.02 
            Negative: 98/100 = 0.98 

        - Total share of people tested positive = (0.0001 * 0.99 +   0.9999 * 0.02) = 0.0201 
        - Total share of people tested positive false 
        = 0.9999 * 0.02 = 0.02 
        - Total share of people tested true positive 
    = 0.99 * 0.0001


        Prob of false positive = Share of False Positive / Total # of Positives = 0.02 / 0.0201 = 0.995 = 99.5%

        Prob of True Positive = Share of True Positive / Total # of positives = (0.99 * 0.0001) / 0.0201  = 0.0049 = 0.5%

    ```

    ```
    Q5. 70% of the population has brown eyes, 30% do not have brown eyes. 60% of the population requires reading glasses, 40% do not need reading glasses. In a city of 10,000 people, how many would both not have brown eyes and not require reading glasses?

    Independent events - just multiply the prob. 
    P(not_brown_eyes) * P(not_require_glasses) = 0.3 * 0.4 = 0.12 

    0.12 * 10000 = 1200 

    ```
    
    ```
    Q6. There are two stacks of cards. Each stack has 4 cards. Each stack has a card with the numbers 1, 2, 3, and 4. There are 16 possible outcomes. You will be allowed to take one card from each stack. Two cards total. What is the probability of drawing at least one card with a 4 from either deck

    Total_num_of_cards = 8
    Total_num_of_possible_outcomes = 4C1 * 4C1 = 16

    = 1/4 * 1 + 1 * 1/4 - 1/16 = 0.4375
    
    ```
    
    ```
    Q7. Suppose you have 3 coins, each with heads on one side and tails on the other. There are 8 possible outcomes. What is the probability that when all three coins are flipped at least 2 coins will result in heads?

    total_num_of_outcomes = 2 * 2 * 2 = 8 


    HHH
    HHT
    HTH
    HTT
    THH
    THT
    TTH
    TTT
	
    ```
    
    ```
    Q8. There are ten people in a class. Ari and Jamaal are twins in this class. At random two people will be chosen as the class representatives. What are the odds that Ari and Jamaal will both be chosen?

    Total_num_of_combinations = 10C2 = 10! / (8! * 2!) = 45 

    Only one combination is Ari and Jammal = 1/45 = 0.0222  
    ```
    
    ```
    Q9. 
    A company has 1000 employees. 70% get the flu vaccine. 95% of those that get the vaccine do NOT get the flu, 5% get the flu. 30% do not get the flu vaccine. 80% of those that do not get the vaccine do not get the flu; 20% that do not get the vaccine do get the flu. How many of the 1000 employees get the vaccine but still get the flu?



    700 Vaccinated
    -- NoFlu: 0.95 * 700 = 665 
    --Flu: 0.05 * 700 = 35 
    300 Non vaccinated
    -- NoFlu: 0.8*300 = 240 
    -- Flu: 0.2 * 300 = 60 
    ```
       
</details>

### Statistics Foundation Notes

<details><summary> Key Points </summary>

<br>

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

    - With the help of CLT, we need not wait for the entire population's data (and the subsequent identification of the population's unknown distribution), we can apply normal distribution principles (like the empirical rule and many more statistical techniques) on the sample means and draw a conclusion about the population

    More about CLT with an example: <br>
    - [Central Limit Theorem’s super power - "You don’t need to know the population distribution"](https://towardsdatascience.com/central-limit-theorem-a-real-life-application-f638657686e1)

        ![](https://upload.wikimedia.org/wikipedia/commons/7/7b/IllustrationCentralTheorem.png)



5. Outlier:
    - Outlier is a relative term. There is no absolute definition (like if a datapoint is 2 or 3 &sigma; away from the mean)

    - How to investigate outliers: <br>
    (one should not simply ignore/remove it) 
        - Is this really an outlier?
        - How did this happen?
        - What can we learn?
        - What needs to change (to make it fit into the distribution)?

</details>
