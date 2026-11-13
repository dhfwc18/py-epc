# Attribution

Py-EPC downloads data from the publishers below. The library is shipped with none of the listed data. Each user
downloads the data they need, and the onus is on the user to follow the respective licence. The licence of Py-EPC's code
does not cover the listed data.

## Energy Performance Certificates, England and Wales

- **Publisher**: Ministry of Housing, Communities and Local Government, through the
  [Get energy performance of buildings data](https://get-energy-performance-data.communities.gov.uk) service.
- **Licence**: every field except the address fields is under the
  [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/), including
  Ordnance Survey UPRNs.
- **Address fields** (`address1` to `address3`, `address`, `postcode`): Ordnance Survey and Royal Mail reserve their
  copyright and database rights. They may be used only for the purposes listed in the
  [licensing restrictions](https://get-energy-performance-data.communities.gov.uk/guidance/licensing-restrictions),
  which relate to the energy performance of buildings. Anyone sharing address data must include a copy of that notice.
- **Personal data**: the data is personal data under UK GDPR and the Data Protection Act 2018. Its users are data
  controllers.

## Energy Performance Certificates, Scotland

- **Publisher**: Scottish Government, through the [Scottish EPC Register](https://www.scottishepcregister.org.uk),
  published on [data.gov.scot](https://data.gov.scot).
- **Licence**: every field except the address fields is under the Open Government Licence v3.0.
- **Address fields** (`ADDRESS1`, `ADDRESS2`, `ADDRESS3` (the post town), `POSTCODE`): Royal Mail copyright, for the
  energy performance purposes listed in each release's publication note. Anyone republishing address data must include
  the Royal Mail copyright notice.

## ONS Postcode Directory

- **Publisher**: Office for National Statistics, through the
  [Open Geography Portal](https://geoportal.statistics.gov.uk).
- **Licence**: Open Government Licence v3.0, for Great Britain, with these attributions:
  1. Contains OS data &copy; Crown copyright and database right [year]
  2. Contains Royal Mail data &copy; Royal Mail copyright and database right [year]
  3. Source: Office for National Statistics licensed under the Open Government Licence v.3.0
- **Northern Ireland**: postcodes starting `BT` are not under the Open Government Licence. Commercial use needs a
  licence from Land and Property Services. Py-EPC always leaves them out.
- See the [ONS licences](https://www.ons.gov.uk/methodology/geography/licences).

## OS Open UPRN

- **Publisher**: Ordnance Survey, through the [OS Data Hub](https://osdatahub.os.uk).
- **Licence**: Open Government Licence v3.0, with this attribution:
  - Contains OS data &copy; Crown copyright and database right [year]
