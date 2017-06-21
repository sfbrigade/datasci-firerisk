<b>San Francisco Fire Risk Project</b></br>

This project attempts to model and acquire data from SF OpenData - and other sources - to predict the relative risk of fire in San Francisco’s buildings and public spaces.

The mapping software will allow the user to type in an address and see fire-related risks and incidences around their area, as well as provide recommendations by fire safety experts in cases where there may be a high enough score to warrant preventive actions.

This project is modeled after Data Science for Social Good's (DSSG) Firebird Project in Atlanta, GA. Consultation is occasionally provided by members of the DSSG and former members of the Atlanta project.

<b>Documentation</b></br>
https://docs.google.com/document/d/1yLQrG6fyxGw2z1n9ikM---qfl7bAh7MoZ2DOMosu_NU/edit

<b>Latest Data Set File (.csv)</b></br>

Can be found in <i>/addresses/</i> folder or downloaded at:
https://drive.google.com/file/d/0B7ce50Tgcva8RnBEU2VTVVlkLWM/view?usp=sharing

<b>Instructions for Submitting a Dataset</b></br>

We are always looking to improve the robustness of our prediction model, so we are always looking for feedback, as well as additional data points and inputs to add to our repository!

1.  Please take a look at the latest address file as a reference for how our addresses are formatted and matched.  We are using addresses from the San Francisco area that is standardized by the Enterprise Addressing System, which should give you multiple options to connect your data with ours.

2.  Please include a short note or documentation on how, what, and why the data set you provided would be a good indicator of fire risk.  The more research the better, since it'll give our data scientists a better idea how to weigh the data points when compiling it into a fire risk score.

3.  The format that we are looking for when we do our merge:

- File Format: .csv
- First Column: EAS BaseID and/or CNN [So we can match address data to our model.]
- The Rest: Column names with data points related to fire risk, multiple columns OK.  [The more complete the data set is, the better!]

4.  Come find us in our Slack channel at https://sfbrigade.slack.com - #datasci-firerisk!  Introduce yourself or contact @ryangtanaka for more details.  Or come join us at Civic Hack Night at with Code for America/San Francisco with the SF Brigade!  http://codeforsanfrancisco.org/

<b>About the Repository</b></br>

<i>/lib/</i> folder contains the model for risk assessment.

<i>address_matching</i> is a prototype Python script that does inexact/fuzzy matching of street address strings to a standardized format.
