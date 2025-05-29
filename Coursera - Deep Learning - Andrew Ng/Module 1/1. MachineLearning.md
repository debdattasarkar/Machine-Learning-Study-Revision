# Machine Learning definition

> **Arthur Samuel (1959):**
> Machine Learning: Field of study that gives computers the ability to learn without being explicitly programmed.

---

# Understanding Machine Learning: Task, Experience, and Performance

## Introduction

This project explores one of the foundational definitions in machine learning introduced by **Tom Mitchell (1998)**, which describes what constitutes a well-posed learning problem. The concept is illustrated using a common real-world scenario: spam email classification.

## Table of Contents

* [Introduction](#introduction)
* [Tom Mitchell's Definition](#tom-mitchells-definition)
* [Learning Framework](#learning-framework)
* [Email Spam Classification Example](#email-spam-classification-example)
* [Explanation](#explanation)
* [Conclusion](#conclusion)

## Tom Mitchell's Definition

> **Tom Mitchell (1998):**
> *A computer program is said to **learn** from experience **E** with respect to some task **T** and some performance measure **P**, if its performance on **T**, as measured by **P**, improves with experience **E***.

This definition forms the backbone of how we understand and structure learning problems in the field of machine learning.

So Tom defines machine learning by saying that a well-posed learning problem is defined as follows. He says, a computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E. I actually think he came out with this definition just to make it rhyme. For the checkers playing examples, the experience E would be the experience of having the program play tens of thousands of games itself. The task T would be the task of playing checkers, and the performance measure P will be the probability that wins the next game of checkers against some new opponent.

## Learning Framework

According to the definition:

* **Task (T):** The objective or problem the system is trying to solve.
* **Experience (E):** The data or interactions from which the system learns.
* **Performance measure (P):** The metric used to evaluate how well the task is being performed.

## Email Spam Classification Example

**Scenario:**
An email program observes user actions (marking emails as spam or not spam) and uses that behavior to improve its spam detection over time.

### Question:

What is the **Task T** in this setting?

### Options:

1. **Classifying emails as spam or not spam** ✅ (Correct - Task **T**)
2. Watching the user label emails (Experience **E**)
3. Measuring accuracy of classification (Performance **P**)
4. None of the above

## Explanation

* **Task (T):** The program's goal — classify an incoming email as either spam or not spam.
* **Experience (E):** Observing how the user handles emails, i.e., labeling them as spam or not.
* **Performance (P):** Accuracy or the fraction of emails correctly classified by the system.

## Conclusion

Understanding the T-E-P framework is essential to structuring any machine learning system. This README presents an intuitive breakdown of Tom Mitchell's definition through a practical email filtering use case. Such clarity helps in designing learning systems with measurable outcomes and clear objectives.

---


# Understanding Machine Learning: Task, Experience, and Performance

## Introduction

This document illustrates a core concept in machine learning: defining a learning problem in terms of **Task (T)**, **Experience (E)**, and **Performance measure (P)**. The example used here is spam email classification, a common use case for machine learning systems.

## Table of Contents

* [Introduction](#introduction)
* [Learning Framework](#learning-framework)
* [Email Spam Classification Example](#email-spam-classification-example)
* [Explanation](#explanation)
* [Conclusion](#conclusion)

## Learning Framework

A computer program is said to **learn from experience E with respect to some task T and some performance measure P** if its performance on T, as measured by P, improves with experience E.

### Key Concepts:

* **Task (T):** What the system is trying to accomplish.
* **Experience (E):** The data or interaction from which the system learns.
* **Performance (P):** How success is measured.

## Email Spam Classification Example

**Scenario:**
An email program observes which emails a user marks as spam or not spam. Over time, it learns to filter spam more accurately based on the user’s behavior.

### Question:

What is the **Task T** in this setting?

### Provided Options:

1. **Classifying emails as spam or not spam** ✅ (Task **T**)
2. Watching you label emails as spam or not spam (Experience **E**)
3. The number (or fraction) of emails correctly classified as spam/not spam (Performance **P**)

## Explanation

* **Task (T):** Classifying emails as spam or not spam.
* **Experience (E):** Observing how the user labels emails.
* **Performance (P):** Accuracy or fraction of correctly classified emails.

The correct task in this setting is **option 1**: *Classifying emails as spam or not spam*.

## Conclusion

This example clarifies how a learning problem is defined in machine learning by identifying the task, the experience the system learns from, and how success is measured. Properly identifying T, E, and P is critical to framing and solving ML problems effectively.

## Machine learning algorithms:
- Supervised learning
- Unsupervised learning
Others: Reinforcement learning, recommender
systems.