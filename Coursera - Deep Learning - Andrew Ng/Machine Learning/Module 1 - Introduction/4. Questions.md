### **1.**

A computer program is said to learn from experience E with
respect to some task T and some performance measure P if its
performance on T, as measured by P, improves with experience E.

Suppose we feed a learning algorithm a lot of historical weather
data, and have it learn to predict weather. What would be a
reasonable choice for P?

* ⭘ The weather prediction task.
* ⭘ None of these.
* ⭘ The process of the algorithm examining a large amount of historical weather data.
* ✅ **The probability of it correctly predicting a future date's weather.**

---

Let me know if you want it formatted in Markdown or included in your README or study material.



This tests your understanding of Tom Mitchell’s definition of a **well-posed learning problem** in machine learning, specifically how to identify the **performance measure (P)**.

---

### 🧠 **Understanding the Learning Problem:**

#### 📘 Tom Mitchell’s Definition Recap:

> A computer program is said to **learn** from **experience E** with respect to some **task T** and some **performance measure P** if its performance on T, as measured by P, improves with experience E.

---

### 🔍 Given Scenario:

* **Experience (E):** Historical weather data.
* **Task (T):** Predicting the weather.
* **Question:** What is the best choice for the **performance measure P**?

---

### ✅ Correct Answer:

> **"The probability of it correctly predicting a future date's weather."**

#### 💡 Why?

This choice reflects how well the algorithm performs the task (weather prediction). Performance measures (P) must quantify **how good** the predictions are — and this option does exactly that by evaluating **accuracy** or **correctness**.

---

### ❌ Incorrect Options Breakdown:

* **"The weather prediction task":** This is the **task (T)**, not the performance measure.
* **"None of these":** Incorrect — one option is clearly valid.
* **"The process of examining data":** This is **experience (E)**, not the performance measure.

---

### ✅ Summary:

* **Task (T):** Predicting weather.
* **Experience (E):** Learning from historical data.
* **Performance (P):** Accuracy or probability of correct prediction.

---

**2.**
Suppose you are working on weather prediction, and use a
learning algorithm to predict tomorrow's temperature (in
degrees Centigrade/Fahrenheit).

Would you treat this as a classification or a regression problem?

* ✅ Regression
* ⭘ Classification

---

### 🧠 Explanation:

This is a **regression** problem because:

* You're predicting a **continuous numeric value** — tomorrow’s temperature (e.g., 21.5°C, 74°F).
* **Regression models** are used when the output is **quantitative** and **not limited to a finite set of classes**.

In contrast, **classification** would be appropriate if the task were to predict a **category** like:

* "Sunny", "Rainy", "Cloudy"
* or "Hot", "Cold", "Moderate"

---

### ✅ Summary:

* **Task:** Predict numeric temperature
* **Type:** **Regression**
* **Why:** The output is **continuous-valued**, not a discrete class label.


---

**3.**
Suppose you are working on stock market prediction, and you
would like to predict the price of a particular stock tomorrow
(measured in dollars). You want to use a learning algorithm for this.

Would you treat this as a classification or a regression problem?

* ⭘ Classification
* ✅ **Regression**

---

### 🧠 Explanation:

This is a **regression problem** because:

* The goal is to predict a **numerical value** — the stock price in dollars.
* Regression algorithms are suitable for predicting **continuous** quantities (e.g., \$123.45, \$178.12).

---

### 🔁 Not Classification Because:

* **Classification** deals with **categorical outputs**, like "rise" vs "fall", or "high", "medium", "low".
* In this question, you're predicting an **exact price**, not a class.

---

### ✅ Summary:

| Task                              | Type       | Why                          |
| --------------------------------- | ---------- | ---------------------------- |
| Predicting tomorrow's stock price | Regression | Output is a continuous value |

Here’s the full **text** extracted from the image along with an **explanation** of which problems are suitable for **supervised learning**:

---

**4.**
Some of the problems below are best addressed using a supervised
learning algorithm, and the others with an unsupervised
learning algorithm. Which of the following would you apply
supervised learning to? (Select all that apply.) In each case, assume some appropriate
dataset is available for your algorithm to learn from.

* ⬜ Examine a large collection of emails that are known to be spam email, to discover if there are sub-types of spam mail.
* ☑ **Given historical data of children's ages and heights, predict children's height as a function of their age.**
* ⬜ Take a collection of 1000 essays written on the US Economy, and find a way to automatically group these essays into a small number of groups of essays that are somehow "similar" or "related".
* ☑ **Given 50 articles written by male authors, and 50 articles written by female authors, learn to predict the gender of a new manuscript's author (when the identity of this author is unknown).**

---

### ✅ Explanation: Supervised vs. Unsupervised

#### ✅ **Supervised Learning (Correct Choices):**

1. **Predict children's height from age**

   * You are given input (age) and output (height) → labeled data.
   * Task: **Regression**

2. **Predict manuscript author gender**

   * You have labeled data: male vs. female.
   * Task: **Classification**

#### ❌ **Unsupervised Learning (Incorrect Choices for Supervised):**

1. **Group emails into sub-types of spam**

   * No labeled categories are given → the model must discover structure itself.
   * Task: **Clustering**

2. **Group similar essays**

   * Again, no predefined labels.
   * Task: **Clustering / Topic Modeling**

---

### 🧾 Summary Table:

| Scenario                   | Type             | Reason                      |
| -------------------------- | ---------------- | --------------------------- |
| Predict height from age    | Supervised (✓)   | Labeled regression task     |
| Predict author gender      | Supervised (✓)   | Labeled classification task |
| Discover sub-types of spam | Unsupervised (✗) | No labels; clustering task  |
| Group similar essays       | Unsupervised (✗) | No labels; clustering task  |

---

**5.**
Which of these is a reasonable definition of machine learning?

* ⭘ Machine learning learns from labeled data.
* ⭘ Machine learning is the science of programming computers.
* ✅ **Machine learning is the field of study that gives computers the ability to learn without being explicitly programmed.**
* ⭘ Machine learning is the field of allowing robots to act intelligently.

---

### 🧠 Explanation:

The **correct answer** is:

> **"Machine learning is the field of study that gives computers the ability to learn without being explicitly programmed."**

This is the **classic and widely accepted definition** by **Arthur Samuel**, one of the pioneers in the field of ML.

---

### ❌ Why the Other Options Are Incorrect:

* **"Learns from labeled data"** – Only applies to **supervised learning**, not all ML.
* **"Science of programming computers"** – Too broad; doesn’t capture the essence of ML.
* **"Field of allowing robots to act intelligently"** – Describes **robotics or AI**, not specifically ML.

---
