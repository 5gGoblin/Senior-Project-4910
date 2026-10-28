# Clean front_suspension, back_suspension and turning_circle_m.
# Input: cars_with_excluded_columns.csv (29,741 rows, 54 columns incl. identifiers)
#
# Suspension: the raw text is a comma-separated list of tags (241 distinct front
# strings, 329 rear). As Gower categoricals these would be nearly useless: Gower
# scores a categorical as match/mismatch, so "independent, mcpherson struts" and
# "independent, mcpherson struts, stabilizer bar" would be fully different.
# We reduce each axle to its layout (a handful of levels) and drop the tags that
# describe components every car has or that are listed inconsistently
# (spring, helical springs, dampers, stabilizer bar).
#
# Turning circle: values already lie in 6.95-16.6 m. We only blank values that
# are physically inconsistent with the wheelbase.

library(dplyr)
library(stringr)
library(readr)

cars <- read_csv("data/cars_with_excluded_columns.csv",
                 show_col_types = FALSE)

# Front "independent, spring" (and similar) names no layout. TRUE sends it to NA
# (Gower skips it); FALSE keeps it as level "independent_other".
FRONT_UNSPECIFIED_TO_NA <- TRUE

has <- function(x, pattern) str_detect(x, regex(pattern, ignore_case = TRUE))

# Front layout. Order matters: the first matching rule wins.
front_layout <- function(x) {
  case_when(
    is.na(x)                                              ~ NA_character_,
    has(x, "\\bdependent|solid axle|drive axle|\\baxle\\b|\\bbeam\\b") &
      !has(x, "\\bindependent")                           ~ "solid_axle",
    has(x, "double wishbone")                             ~ "double_wishbone",
    has(x, "multi wishbone|multi-link")                   ~ "multilink",
    has(x, "mcpherson|macpherson|\\bstrut")               ~ "strut",
    has(x, "wishbone|lever|trailing arm|torsion")         ~ "independent_other",
    has(x, "\\bindependent") & !FRONT_UNSPECIFIED_TO_NA   ~ "independent_other",
    TRUE                                                  ~ NA_character_  # e.g. "spring", "stabilizer bar" only
  )
}

# Rear layout.
rear_layout <- function(x) {
  case_when(
    is.na(x)                                              ~ NA_character_,
    has(x, "semi-dependent|semi-independent|torsion beam|h-shaped beam|twisted beam|cross beam") ~ "torsion_beam",
    has(x, "\\btorsion\\b") & !has(x, "\\bindependent")   ~ "torsion_beam",
    has(x, "de-dion")                                     ~ "solid_axle",
    has(x, "\\bdependent|solid axle|drive axle|\\baxle\\b|leaf spring|panhard|\\bbeam\\b") ~ "solid_axle",
    has(x, "double wishbone")                             ~ "double_wishbone",
    has(x, "multi wishbone|multi-link|trapezoidal wishbone") ~ "multilink",
    has(x, "mcpherson|\\bstrut")                          ~ "strut",
    has(x, "trailing arm")                                ~ "trailing_arm",
    has(x, "\\bindependent|wishbone|lever|rod")           ~ "independent_other",
    TRUE                                                  ~ NA_character_  # "spring", "helical springs", "dampers" only
  )
}

cars_clean <- cars %>%
  mutate(
    # Wheelbase typos found while checking turning circle (published specs)
    wheelbase_mm = case_when(
      Make == "Citroen" & Modle == "DS5"   & wheelbase_mm == 2127 ~ 2727,
      Make == "Changan" & Modle == "Benni" & wheelbase_mm == 1365 ~ NA_real_,
      TRUE ~ wheelbase_mm
    )
  ) %>%
  mutate(
    front_susp_layout = front_layout(front_suspension),
    rear_susp_layout  = rear_layout(back_suspension),
    # Turning circle diameter / wheelbase is 3.3-5.0 for 99.8% of rows.
    # Outside 3.0-5.6 the pair cannot both be right, so blank the turning circle.
    tc_wb_ratio = turning_circle_m / (wheelbase_mm / 1000),
    # Exception: the London taxi (LTI TX) really has an 8 m turning circle.
    turning_circle_m = if_else(!is.na(tc_wb_ratio) &
                                 (tc_wb_ratio < 3.0 | tc_wb_ratio > 5.6) &
                                 Make != "LTI",
                               NA_real_, turning_circle_m)
  ) %>%
  select(-tc_wb_ratio, -front_suspension, -back_suspension,
         -payload_kg, -city_fuel_per_100km_l, -highway_fuel_per_100km_l, -pct_filled) %>%
  mutate(across(c(front_susp_layout, rear_susp_layout), factor))

# Checks
count(cars_clean, front_susp_layout, sort = TRUE)
count(cars_clean, rear_susp_layout,  sort = TRUE)
summary(cars_clean$turning_circle_m)

# Raw strings that fell to NA, to review by hand
cars %>%
  mutate(f = front_layout(front_suspension)) %>%
  filter(is.na(f), !is.na(front_suspension)) %>%
  count(front_suspension, sort = TRUE)
cars %>%
  mutate(r = rear_layout(back_suspension)) %>%
  filter(is.na(r), !is.na(back_suspension)) %>%
  count(back_suspension, sort = TRUE)

# Full clean data, then the split for Gower. Both split files are sorted by id_trim and keep id_trim, so row r of
# one is row r of the other. These are the columns gower_prep.py excludes by
# default, plus mixed_fuel_estimated (a flag saying the value was imputed).
id_cols <- c("id_trim", "Make", "Modle", "Generation", "Year_from", "Year_to",
             "Series", "Trim", "Body_type", "emission_standards",
             "mixed_fuel_estimated")

cars_clean <- cars_clean %>% select(-any_of("...1"), -any_of("Unnamed: 0")) %>% arrange(id_trim)
write_csv(cars_clean, "data/cars_clean.csv", na = "")
saveRDS(cars_clean, "data/cars_clean.rds")
write_csv(select(cars_clean, all_of(id_cols)), "data/cars_clean_identifiers.csv", na = "")
write_csv(select(cars_clean, id_trim, !all_of(id_cols)), "data/cars_clean_specs.csv", na = "")
