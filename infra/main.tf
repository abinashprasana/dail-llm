terraform {
  required_version = ">= 1.9.0, < 2.0.0"
  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "5.25.0"
    }
  }
}

provider "cloudflare" {}

variable "cloudflare_account_id" {
  description = "Cloudflare account for free D1 year shards. Set TF_VAR_cloudflare_account_id."
  type        = string
}

variable "first_year" {
  type    = number
  default = 2014
}

variable "last_year" {
  type    = number
  default = 2026
}

variable "pilot_start_year" {
  description = "Optional two-year shard start for a single-database sizing pilot."
  type        = number
  default     = null
}

variable "existing_pilot_database_id" {
  description = "Existing 2022-2023 D1 pilot, created outside Terraform. Terraform references but does not manage it."
  type        = string
  default     = "ec1bb53b-f607-4c81-8294-9683f4d8800d"
}

locals {
  shard_start_years = range(var.first_year, var.last_year + 1, 2)
}

check "free_d1_database_limit" {
  assert {
    condition     = var.first_year <= var.last_year && ceil((var.last_year - var.first_year + 1) / 2) <= 10
    error_message = "Workers Free allows at most 10 D1 databases; use a valid range of no more than 20 years."
  }
}

check "pilot_shard" {
  assert {
    condition     = var.pilot_start_year == null || contains(local.shard_start_years, var.pilot_start_year)
    error_message = "pilot_start_year must be the first year of one configured two-year shard."
  }
}

resource "cloudflare_d1_database" "debates" {
  # Two consecutive years per database keep the 2014-2026 corpus below the
  # Workers Free limit of ten databases. Actual D1 sizes still need measuring.
  for_each = toset([for year in local.shard_start_years : tostring(year)
    if year != 2022 && (var.pilot_start_year == null || year == var.pilot_start_year)
  ])
  account_id   = var.cloudflare_account_id
  name         = "dail-debates-${each.key}-${tonumber(each.key) + 1}"
  jurisdiction = "eu"

  lifecycle {
    prevent_destroy = true
  }
}

output "d1_database_ids" {
  description = "Year-to-ID map for DAIL_D1_DATABASES. The existing 2022-2023 pilot is referenced, not managed."
  value = merge([for start in local.shard_start_years : {
    for year in range(start, min(start + 2, var.last_year + 1)) :
    tostring(year) => start == 2022 ? var.existing_pilot_database_id : cloudflare_d1_database.debates[tostring(start)].id
    if var.pilot_start_year == null || start == var.pilot_start_year
  }]...)
}
