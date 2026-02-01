# Phishing Website Detection Using Machine Learning

---

## 1. Executive Summary

Phishing websites are fraudulent online pages designed to steal user credentials, financial data, and personal information. As digital transactions increase, phishing attacks pose serious risks to individuals and organizations. This project aims to develop an intelligent machine learning system that detects phishing websites using URL structure, domain properties, and webpage behavior features.

The dataset contains 11,430 websites with 89 features describing URL characteristics, domain information, and webpage security signals. The goal is to classify websites as **phishing** or **legitimate**, helping organizations prevent cyber fraud, protect user trust, and strengthen cybersecurity systems.

---

## 2. Business Problem Understanding

### 2.1 Problem Statement
Cyber attackers create fake websites that closely resemble legitimate platforms to trick users into revealing sensitive information. Traditional blacklist methods cannot detect newly created phishing websites. A predictive detection system is needed to identify phishing websites automatically before users interact with them.

### 2.2 Scope
- Real-time phishing detection
- Browser/email security integration
- Fraud prevention systems
- Automated cybersecurity monitoring

### 2.3 Importance of Detecting Phishing Websites
- Prevents financial loss and fraud
- Protects user credentials and privacy
- Reduces brand reputation damage
- Strengthens enterprise security infrastructure

---

## 3. Literature Review Insights

### Common Phishing Characteristics
- Long and complex URLs
- Use of IP address instead of domain
- Excessive special characters
- Multiple subdomains
- Suspicious redirects
- Fake HTTPS indicators
- Short domain age
- External scripts and media links

### Detection Challenges
- Phishing domains have short lifespans
- URL obfuscation techniques
- Visual similarity to legitimate sites
- Evolving attack patterns

### Potential Solutions
- URL-based machine learning models
- Domain registration analysis
- Webpage behavior analysis
- Hybrid ML + blacklist systems

---

# Dataset Exploration Report

---

## 4. Dataset Overview

| Attribute | Value |
|----------|------|
| Total Records | 11,430 websites |
| Total Features | 89 columns |
| Feature Types | 74 Integer, 13 Float, 2 Object |
| Target Variable | `status` (phishing / legitimate) |

### Target Variable Distribution
| Class | Count |
|------|------|
| Phishing | 5,715 |
| Legitimate | 5,715 |

✔ Dataset is **perfectly balanced**.

---

## 5. Feature Categories

## Feature Description and Relevance to Phishing Detection

| Feature | Description | Relevance to Phishing Detection |
|---------|-------------|--------------------------------|
| length_url | Total number of characters in the URL | Phishing URLs are often very long to hide suspicious parts |
| nb_dots | Number of dots in URL | Many dots indicate multiple subdomains, common in phishing |
| nb_hyphens | Count of '-' symbols in URL | Attackers use hyphens to mimic real domains |
| nb_at | Presence of '@' symbol | '@' redirects browser and hides actual domain |
| nb_slash | Number of '/' in URL | Excessive slashes suggest deep fake directory paths |
| ratio_digits_url | Ratio of digits in URL | Phishing URLs often contain random numbers |
| domain_age | Age of the domain (in days) | Phishing domains are newly registered |
| domain_registration_length | Duration of domain registration | Legitimate domains are registered for longer periods |
| whois_registered_domain | WHOIS registration status | Unregistered or hidden WHOIS info is suspicious |
| dns_record | Availability of DNS record | Missing DNS record suggests malicious intent |
| google_index | Whether the site is indexed by Google | Phishing sites are often not indexed |
| page_rank | Website ranking score | Low-ranked domains are usually untrustworthy |
| https_token | Presence of "https" token in domain | Fake domains misuse https token to appear secure |
| ssl_final_state | SSL certificate trust status | Invalid SSL often indicates phishing |
| nb_redirection | Number of redirections | Phishing pages use multiple redirects to hide origin |
| nb_external_redirection | Redirections to external domains | Used to send users to malicious pages |
| ratio_extHyperlinks | Ratio of external hyperlinks | Phishing pages link to external malicious sources |
| ratio_extMedia | Ratio of external media objects | External media is common in phishing pages |
| login_form | Presence of login form | Used to steal user credentials |
| popup_window | Presence of pop-up windows | Phishing uses popups to request sensitive info |
| iframe | Use of invisible iframe | Used to load malicious content secretly |
| submit_email | Email submission form | Phishing collects credentials via email forms |
| external_favicon | Favicon loaded from external source | Indicates fake website branding |
| links_in_tags | Hyperlinks inside HTML tags | Used to hide malicious redirections |
| ratio_intHyperlinks | Ratio of internal hyperlinks | Phishing sites often have fewer internal links |
| length_hostname | Length of hostname | Long hostnames indicate suspicious domains |
| nb_www | Count of 'www' in URL | Multiple 'www' used to confuse users |
| nb_com | Count of 'com' in URL | Fake domains often repeat 'com' |
| shortest_word_host | Shortest word length in hostname | Random short words common in phishing domains |
| longest_word_path | Longest word in URL path | Long random paths are suspicious |
| avg_word_path | Average word length in path | Irregular path words indicate phishing |
| char_repeat | Repeated characters in URL | Used to mimic legitimate sites |
| ratio_digits_host | Ratio of digits in hostname | Numeric hostnames often malicious |
| nb_subdomains | Number of subdomains | Many subdomains used to appear legitimate |
| tld_length | Length of top-level domain | Unusual TLD lengths linked with phishing |
| domain_in_title | Domain present in page title | Mismatch indicates phishing |
| domain_with_copyright | Domain in copyright text | Absence suggests fake website |
| nb_hyperlinks | Total hyperlinks on page | Phishing pages have fewer meaningful links |
| nb_images | Number of images | Low image count suggests fake site |
| nb_css | Number of CSS files | Minimal styling in phishing pages |
| nb_js | Number of JavaScript files | Excess scripts used for malicious actions |
| nb_self_ref | Self-referencing links | Legit sites link internally more |
| nb_empty_links | Empty or broken links | Poor site quality often phishing |
| nb_internal_links | Internal hyperlinks count | Legit sites have structured internal links |
| nb_external_links | External links count | Phishing uses more external redirects |
| status | Target variable (phishing/legitimate) | Classification output |


### URL Structure Features
Examples: `length_url`, `nb_dots`, `nb_hyphens`, `nb_at`, `nb_slash`, `ratio_digits_url`  
➡ Long URLs and excessive symbols often indicate phishing.

### Domain-Based Features
Examples: `domain_age`, `domain_registration_length`, `whois_registered_domain`, `dns_record`, `page_rank`  
➡ Phishing domains are usually newly registered and poorly ranked.

### Security Indicators
Examples: `https_token`, `ssl_final_state`, `google_index`  
➡ Fake HTTPS tokens or unindexed domains suggest risk.

### Webpage Behavior Features
Examples: `popup_window`, `iframe`, `login_form`, `external_favicon`, `submit_email`  
➡ Phishing pages often request credentials via pop-ups and forms.

### Link & Redirection Features
Examples: `nb_redirection`, `nb_external_redirection`, `ratio_extHyperlinks`  
➡ Excess redirection is a common phishing tactic.

---

## 6. Exploratory Data Analysis (EDA)

### Key Observations

- Phishing URLs tend to have **higher URL length**.
- Higher number of special characters correlates with phishing.
- Domains with **low age** are more likely phishing.
- Websites with **many external hyperlinks** are often malicious.
- Use of IP address in URL is strongly linked to phishing.
- Redirect counts are higher in phishing samples.

### Correlation Patterns
- URL length and digit ratio show positive correlation with phishing.
- Domain age negatively correlates with phishing.
- External media and redirection features show strong importance.

---

## 7. Data Quality Assessment

| Aspect | Status |
|--------|--------|
| Missing Values | None detected |
| Duplicate Records | Not significant |
| Data Imbalance | None (balanced dataset) |
| Outliers | Present in URL length and traffic features |

### Handling Strategy
- Outliers handled carefully using scaling.
- Feature selection applied to reduce redundancy.

---

## 8. Key Insights from Data

- URL-based features alone carry strong predictive power.
- Domain age and registration length are highly important.
- Webpage behavioral features improve detection accuracy.
- Multi-feature analysis outperforms blacklist systems.

---

## 9. Business Value of This Project

- Early detection of phishing threats
- Integration into browsers and firewalls
- Protection against identity theft
- Reduced financial fraud
- Enhanced customer trust and cybersecurity posture

---

## 10. Next Phase

- Model development and comparison
- Hyperparameter tuning
- Feature importance analysis
- Deployment as API or browser plugin

---

## 11. Tools & Technologies

| Category | Tools |
|---------|------|
| Language | Python |
| Libraries | Pandas, NumPy, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| ML Models (Planned) | Logistic Regression, Random Forest, XGBoost |

---

## 12. Author

Name: **Shravan Shidruk (#37015)**<br>
GitHub : **https://github.com/shravanshidruk16/ITVedant**<br>
Course: **Code B - Data Science Integrated Internship**<br>
Week : **Week 1: Understanding the Problem and Dataset**<br>
Institution: **ITVedant**

---
