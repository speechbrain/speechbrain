#!/usr/bin/env perl

# Copyright 2020 Xiaomi Corporation (Author: Junbo Zhang)
# Apache 2.0

use strict;
use warnings;

my $Usage = <<EOU;
fstaddselfloops.pl:
Adds self-loops to states of an FST to propagate disambiguation symbols through it.
They are added on each final state and each state with non-epsilon output symbols
on at least one arc out of the state. 

Usage: local/fstaddselfloops.pl <wdisambig_phone> <wdisambig_word> < <openfst_text>
 e.g.: cat L_disambig.txt | local/fstaddselfloops.pl 347 200004 > L_disambig_with_loop.txt
EOU

if (@ARGV != 2) {
  die $Usage;
}

my $wdisambig_phone = shift @ARGV;
my $wdisambig_word = shift @ARGV;

my %states_needs_self_loops;
while (<>) {
    print $_;

    my @items = split(/\s+/);
    if (@items == 2) {
        # it is a final state
        $states_needs_self_loops{$items[0]} = 1;
    } elsif (@items == 5) {
        my ($src, $dst, $inlabel, $outlabel, $score) = @items;
        $states_needs_self_loops{$src} = 1 if ($outlabel != 0);
    } else {
        die "Invalid openfst line.";
    }
}

foreach (keys %states_needs_self_loops) {
    print "$_ $_ $wdisambig_phone $wdisambig_word 0.0\n"
}
