terraform {
  required_version = ">= 1.9.0, < 2.0.0"
  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "5.25.0"
    }
    vercel = {
      source  = "vercel/vercel"
      version = "5.17.1"
    }
  }
}

provider "cloudflare" {}
provider "vercel" {}

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

variable "vercel_project_name" {
  type    = string
  default = "dail-llm"
}

variable "inspect_existing_vercel_project" {
  type    = bool
  default = false
}

data "vercel_project" "existing" {
  count = var.inspect_existing_vercel_project ? 1 : 0
  name  = var.vercel_project_name
}

resource "cloudflare_d1_database" "debates" {
  for_each     = toset([for year in range(var.first_year, var.last_year + 1) : tostring(year)])
  account_id   = var.cloudflare_account_id
  name         = "dail-debates-${each.key}"
  jurisdiction = "eu"

  lifecycle {
    prevent_destroy = true
  }
}

output "d1_database_ids" {
  description = "Year-to-ID map for DAIL_D1_DATABASES after a reviewed apply."
  value       = { for year, database in cloudflare_d1_database.debates : year => database.id }
}

output "existing_vercel_project_id" {
  value = var.inspect_existing_vercel_project ? data.vercel_project.existing[0].id : null
}
