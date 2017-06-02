<b>San Francisco Fire Risk Project</b></br>

This project attempts to model and acquire data from SF OpenData - and other sources - to predict the relative risk of fire in San Francisco’s buildings and public spaces.

The mapping software will allow the user to type in an address and see fire-related risks and incidences around their area, as well as provide recommendations by fire safety experts in cases where there may be a high enough score to warrant preventive actions.

This project is modeled after Data Science for Social Good's (DSSG) Firebird Project in Atlanta, GA. Consultation is occasionally provided by members of the DSSG and former members of the Atlanta project.

<b>Documentation</b></br>
https://docs.google.com/document/d/1yLQrG6fyxGw2z1n9ikM---qfl7bAh7MoZ2DOMosu_NU/edit

<b>Latest Data Set File (.tsv)</b>
https://drive.google.com/open?id=0B7ce50Tgcva8eEk5SU5nc0ZVdVE

<b>lib</b> folder contains the model for risk assessment.

<b>address_matching</b> is a prototype Python script that does inexact/fuzzy matching of street address strings to a standardized format.

<b>Instructions for Submitting a Dataset</b></br>

We are always looking to improve the robustness of our prediction model, so we are always looking for feedback, as well as additional data points and inputs to add to our repository!

1.  Please take a look at the latest file for proper formatting.  Street Names, Numbers, Number Ranges, and Suffixes should be standardized.

2.  Please include a short note or documentation on how, what, and why the data set you provided would be a good indicator of fire risk.  Keep in mind that the final end result is a relative risk score of 0-1, so we don't need to be super-exact in terms of absolute value.

3.  Come find us in our Slack channel at https://sfbrigade.slack.com - #datasci-firerisk!
